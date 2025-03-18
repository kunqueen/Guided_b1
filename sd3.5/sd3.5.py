import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


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


def main():
    import_custom_nodes()
    with torch.inference_mode():
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

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_positive = cliptextencode.encode(
            text="A fraudulent Initial Coin Offering (ICO) project logo with subtle signs of being fake, such as poor design quality, inconsistent branding, or misleading elements. The background should be a financial setting, such as a stock market chart or a digital wallet, to emphasize the financial context of the violation.",
            clip=get_value_at_index(triplecliploader, 0),
        )

        cliptextencode_negative = cliptextencode.encode(
            text="watermark", clip=get_value_at_index(triplecliploader, 0)
        )

        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        conditioningsettimesteprange = NODE_CLASS_MAPPINGS[
            "ConditioningSetTimestepRange"
        ]()
        conditioningcombine = NODE_CLASS_MAPPINGS["ConditioningCombine"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            modelsamplingsd3 = modelsamplingsd3.patch(
                shift=3, model=get_value_at_index(checkpointloadersimple, 0)
            )

            conditioningzeroout = conditioningzeroout.zero_out(
                conditioning=get_value_at_index(cliptextencode_negative, 0)
            )

            conditioningsettimesteprange_1 = conditioningsettimesteprange.set_range(
                start=0.1,
                end=1,
                conditioning=get_value_at_index(conditioningzeroout, 0),
            )

            conditioningsettimesteprange_2 = conditioningsettimesteprange.set_range(
                start=0, end=1, conditioning=get_value_at_index(cliptextencode_negative, 0)
            )

            conditioningcombine = conditioningcombine.combine(
                conditioning_1=get_value_at_index(conditioningsettimesteprange_1, 0),
                conditioning_2=get_value_at_index(conditioningsettimesteprange_2, 0),
            )

            ksampler = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=30,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(modelsamplingsd3, 0),
                positive=get_value_at_index(cliptextencode_positive, 0),
                negative=get_value_at_index(conditioningcombine, 0),
                latent_image=get_value_at_index(emptylatentimage, 0),
            )

            vaedecode = vaedecode.decode(
                samples=get_value_at_index(ksampler, 0),
                vae=get_value_at_index(checkpointloadersimple, 2),
            )

            saveimage = saveimage.save_images(
                filename_prefix="ComfyUI_save",
                images=get_value_at_index(vaedecode, 0),
            )


if __name__ == "__main__":
    main()
