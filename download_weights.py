import os
import subprocess

def main():
    # Create weights directory
    os.makedirs('weights', exist_ok=True)
    
    # YOLOv3 weights URL
    url = 'https://pjreddie.com/media/files/yolov3.weights'
    output_path = 'weights/yolov3.weights'
    
    print("Downloading YOLOv3 weights...")
    try:
        subprocess.run(['wget', '-O', output_path, url], check=True)
        print("\nDownload complete! Weights saved in weights/yolov3.weights")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading weights: {e}")
    except FileNotFoundError:
        print("wget is not installed. Please install it or download the weights manually from:")
        print(url)

if __name__ == '__main__':
    main()
