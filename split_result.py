import json
import os
from collections import defaultdict
from pathlib import Path

class ResultSplitter:
    def __init__(self, results_file, output_dir):
        self.results_file = results_file
        self.output_dir = output_dir
        
        # Mapping for identity names in filenames
        self.identity_filename_map = {
            "age_18_to_21": "18_21",
            "age_22_to_25": "22_25",
            "age_26_to_29": "26_29",
            "age_30_to_34": "30_34",
            "age_35_to_40": "35_40",
            "JuniorCollege": "JuniorCollege",
            "JuniorHigh": "JuniorHigh",
            "SeniorHigh": "SeniorHigh",
            "TechnicalSecondarySchool": "TechnicalSecondary",
            "University": "University",
            "female": "female",
            "male": "male",
            "id_free": "no_id"
        }
        
        # Mapping for aspect names in filenames
        self.aspect_filename_map = {
            "quality": "perception",
            "aesthetic": "aesthetic",
            "emotion": "empathy"
        }

    def load_results(self):
        with open(self.results_file, 'r') as f:
            return json.load(f)

    def create_output_structure(self):
        """Create output directory structure"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def split_results(self):
        """Split results into separate files by identity and aspect"""
        results = self.load_results()
        self.create_output_structure()

        # Initialize defaultdict for each identity-aspect combination
        split_data = defaultdict(lambda: defaultdict(dict))

        # Process all results
        for image_name, analysis in results.items():
            # Remove .jpg extension if present
            image_name = image_name.replace('.jpg', '')

            # Skip if analysis is not valid
            if not isinstance(analysis, dict):
                continue

            # For each identity and aspect combination
            for identity, identity_filename in self.identity_filename_map.items():
                if identity in analysis:
                    identity_data = analysis[identity]
                    for aspect, aspect_filename in self.aspect_filename_map.items():
                        if aspect in identity_data:
                            value = identity_data[aspect]
                            # Add semicolon to match the format
                            split_data[identity][aspect][image_name] = f"{value};"

        # Save separate files for each combination
        for identity, identity_filename in self.identity_filename_map.items():
            for aspect, aspect_filename in self.aspect_filename_map.items():
                filename = f"{aspect_filename}_{identity_filename}.json"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save to file with the simple key-value format
                with open(filepath, 'w') as f:
                    json.dump(split_data[identity][aspect], f, indent=2)
                
                print(f"Created: {filename} with {len(split_data[identity][aspect])} entries")

    def check_missing_images(self):
        """Check which images are missing for each identity-aspect combination"""
        results = self.load_results()
        
        # Get all unique image names
        all_images = set()
        for image_name in results.keys():
            all_images.add(image_name.replace('.jpg', ''))
        
        # Check missing images for each combination
        missing_data = defaultdict(lambda: defaultdict(list))
        
        for identity in self.identity_filename_map.keys():
            for aspect in self.aspect_filename_map.keys():
                for image_name in all_images:
                    # Check if image exists in results
                    if (image_name + '.jpg') in results:
                        analysis = results[image_name + '.jpg']
                        if not isinstance(analysis, dict):
                            missing_data[identity][aspect].append(image_name)
                            continue
                            
                        if identity not in analysis:
                            missing_data[identity][aspect].append(image_name)
                        elif aspect not in analysis[identity]:
                            missing_data[identity][aspect].append(image_name)
        
        # Print missing images report
        for identity in self.identity_filename_map.keys():
            for aspect in self.aspect_filename_map.keys():
                missing = missing_data[identity][aspect]
                if missing:
                    print(f"\nMissing images for {identity} - {aspect}:")
                    print(f"Total missing: {len(missing)}")
                    print("Missing images:", missing[:10], "..." if len(missing) > 10 else "")

def main():
    results_file = "process/claude_normalized_results.json"
    output_dir = "process/claude_restructured_result"
    
    splitter = ResultSplitter(results_file, output_dir)
    print("\nChecking missing images...")
    splitter.check_missing_images()
    
    splitter.split_results()
    
    print("\nSplit complete!")
    print(f"Total files created: {len(splitter.identity_filename_map) * len(splitter.aspect_filename_map)}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main() 
