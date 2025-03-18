import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import json
from pathlib import Path


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def load_prompts(json_path):
    """Load prompts from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_filename_mapping(mapping, output_dir):
    """Save filename to key mapping as JSON.
    
    Args:
        mapping (dict): Dictionary mapping filenames to original keys
        output_dir (str): Directory to save the mapping file
    """
    mapping_file = Path(output_dir) / 'filename_mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def main():
    import_custom_nodes()
    
    # Load prompts from JSON
    prompts = load_prompts('output.json')
    
    # Create output directory
    output_dir = 'generated_images'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Dictionary to store filename mappings
    filename_mapping = {}
    
    with torch.inference_mode():
        # Initialize models (only need to do this once)
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd3.5_medium.safetensors"
        )

        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage = emptylatentimage.generate(
            width=512, height=512, batch_size=1
        )

        triplecliploader = NODE_CLASS_MAPPINGS["TripleCLIPLoader"]()
        triplecliploader = triplecliploader.load_clip(
            clip_name1="clip_g.safetensors",
            clip_name2="clip_l.safetensors",
            clip_name3="t5xxl_fp16.safetensors",
        )

        # Initialize other nodes
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        conditioningsettimesteprange = NODE_CLASS_MAPPINGS["ConditioningSetTimestepRange"]()
        conditioningcombine = NODE_CLASS_MAPPINGS["ConditioningCombine"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        # Generate images for each prompt
        for idx, (key, data) in enumerate(prompts.items(), 1):
            prompt = data.get('prompt', '').strip('"')  # Remove quotes if present
            if not prompt:
                continue
                
            print(f"Generating image {idx}/{len(prompts)}: {key}")
            
            # Create numbered filename and store mapping
            filename = f"image_{idx:04d}"  # Creates filenames like image_0001, image_0002, etc.
            filename_mapping[filename] = {
                'key': key,
                'prompt': prompt
            }
            
            # Encode text
            cliptextencode_positive = cliptextencode.encode(
                text=prompt,
                clip=get_value_at_index(triplecliploader, 0),
            )

            cliptextencode_negative = cliptextencode.encode(
                text="watermark",
                clip=get_value_at_index(triplecliploader, 0),
            )

            # Rest of the pipeline
            modelsamplingsd3_output = modelsamplingsd3.patch(
                shift=3,
                model=get_value_at_index(checkpointloadersimple, 0),
            )

            conditioningzeroout_output = conditioningzeroout.zero_out(
                conditioning=get_value_at_index(cliptextencode_negative, 0),
            )

            conditioningsettimesteprange_1 = conditioningsettimesteprange.set_range(
                start=0.1,
                end=1,
                conditioning=get_value_at_index(conditioningzeroout_output, 0),
            )

            conditioningsettimesteprange_2 = conditioningsettimesteprange.set_range(
                start=0,
                end=1,
                conditioning=get_value_at_index(cliptextencode_negative, 0),
            )

            conditioningcombine_output = conditioningcombine.combine(
                conditioning_1=get_value_at_index(conditioningsettimesteprange_1, 0),
                conditioning_2=get_value_at_index(conditioningsettimesteprange_2, 0),
            )

            ksampler_output = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=30,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(modelsamplingsd3_output, 0),
                positive=get_value_at_index(cliptextencode_positive, 0),
                negative=get_value_at_index(conditioningcombine_output, 0),
                latent_image=get_value_at_index(emptylatentimage, 0),
            )

            vaedecode_output = vaedecode.decode(
                samples=get_value_at_index(ksampler_output, 0),
                vae=get_value_at_index(checkpointloadersimple, 2),
            )

            # Save image with numbered filename
            saveimage.save_images(
                filename_prefix=str(Path(output_dir) / filename),
                images=get_value_at_index(vaedecode_output, 0),
            )
            
        # Save the filename mapping
        save_filename_mapping(filename_mapping, output_dir)


if __name__ == "__main__":
    main()
