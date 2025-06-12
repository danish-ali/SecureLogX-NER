import os

# Look for any folders that might contain the model
folders = [f for f in os.listdir('.') if os.path.isdir(f)]
print("Folders in current directory:", folders)
