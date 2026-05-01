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
                    if stripped.startswith('#') and not stripped.startswith('#!'):
                        continue
                    
                    if '
                        line = line.split('
                        
                    new_lines.append(line)
                
                with open(path, 'w') as f:
                    f.writelines(new_lines)

if __name__ == "__main__":
    print("Stripping comments from code directory...")
    remove_comments('code')
    print("Done!")
