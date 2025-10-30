import modal

# Define Modal image with all data science dependencies
# Note: This image will be built once and cached by Modal
image = modal.Image.debian_slim(python_version="3.11")\
    .pip_install_from_requirements("backend/modal_runtime/requirements.txt")\
    .add_local_file("backend/modal_runtime/driver.py", "/root/driver.py")

# Use App.lookup to create/find the app lazily
# This will create the app if it doesn't exist, or find it if it does
app = modal.App.lookup("lg-urban-executor", create_if_missing=True)
