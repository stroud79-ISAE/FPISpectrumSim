import os

def list_files_to_txt(directory, output_file, use_relative=True):
    """
    Lists all file paths in the specified directory and writes them to a text file.
    
    Args:
        directory (str): The path to the directory to scan
        output_file (str): The path to the output text file
        use_relative (bool): If True, paths are relative to the script's directory; if False, absolute paths
    """
    try:
        # Get the directory of the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Convert input directory to absolute path for consistency
        directory = os.path.abspath(directory)
        
        # Verify directory exists
        if not os.path.isdir(directory):
            raise ValueError(f"Directory does not exist: {directory}")
        
        # Open the output file in write mode
        with open(output_file, 'w', encoding='utf-8') as f:

            # Walk through the directory
            for root, _, files in os.walk(directory):
                for file in files:

                    # Construct full file path
                    file_path = os.path.join("/rds/projects/r/reidb-waccm-x/tiegcm/run005/", file)

                    # Convert to relative path if requested
                    if use_relative:
                        file_path = os.path.relpath(file_path, script_dir)

                    # Write the file path to the output file
                    f.write(file_path + '\n')

        print(f"File paths successfully written to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":

    # Specify the output file name
    output_txt = "files_to_process.txt"
    directory = ".../tiegcm/run005/" # change to your directory
    
    # Call the function with provided directory and path preference
    list_files_to_txt(directory, output_txt, use_relative=False)

