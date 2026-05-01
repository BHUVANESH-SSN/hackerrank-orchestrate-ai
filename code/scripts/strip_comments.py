import os

def remove_comments(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    stripped = line.lstrip()
                    # Skip full-line comments, but keep the python shebang
                    if stripped.startswith('#') and not stripped.startswith('#!'):
                        continue
                    
                    # Handle inline comments safely (only if '#' is preceded by a space)
                    # This avoids breaking strings that contain URLs or hex colors like '#FFFFFF'
                    if ' #' in line:
                        line = line.split(' #')[0] + '\n'
                        
                    new_lines.append(line)
                
                with open(path, 'w') as f:
                    f.writelines(new_lines)

if __name__ == "__main__":
    print("Stripping comments from code directory...")
    remove_comments('code')
    print("Done!")
