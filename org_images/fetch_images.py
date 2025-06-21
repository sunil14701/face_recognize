import json
import os
import requests

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)    
    return data

def download_image(url, folder_path, filename=None, email_id=None):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Get the image content
    response = requests.get(url)
    if response.status_code == 200:
        # Use the last part of the URL if no filename is provided
        if not filename:
            filename = url.split("/")[-1]
        each_user_folder = os.path.join(folder_path, email_id)
        os.makedirs(each_user_folder, exist_ok=True)
        file_path = os.path.join(each_user_folder, filename)

        
        # Write the image to the file
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Image saved to {file_path}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")


if __name__ == "__main__":
    file_path = 'emp_data.json'
    data = read_json_file(file_path)
    id = 0
    if data is not None:
        for emp_dict in data:
            emp_name = emp_dict.get('EmployeeName')
            img_url = emp_dict.get('ImagePath')
            Designation = emp_dict.get('Designation')
            email_id = emp_dict.get('EmailId')
            download_image(img_url,'Emp_images', f"{emp_name}.jpg", email_id)
            

            # Create folder for each employee using their email ID
            each_user_folder = os.path.join('Emp_images', email_id)
            os.makedirs(each_user_folder, exist_ok=True)

            # Save employee JSON data
            file_path = os.path.join(each_user_folder, "data.json")
            with open(file_path, "w") as f:
                json.dump(emp_dict, f, indent=4)
            
            id += 1
    
    print("Total Employees Processed:", id)
    