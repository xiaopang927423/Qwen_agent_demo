import os
def take_screenshot_and_save(device, local_directory,screenshot_name):
    # 设备上的截图路径
    device_path = "/sdcard/screenshot.png"

    # 本地保存截图的路径
    local_path = os.path.join(local_directory, screenshot_name)

    try:
        # 在设备上截屏并保存到 /sdcard/screenshot.png
        print("正在通过adb截图...")
        device.shell(f"screencap -p {device_path}")

        # 将截图从设备拉取到本地目录
        print(f"将图片保存到 {local_path}...")
        device.pull(device_path, local_path)

        print(f"成功保存到 {local_path}")

        # 可选：删除设备上的截图以节省空间
        print("正在清除手机上的截图...")
        device.shell(f"rm {device_path}")

    except Exception as e:
        print(f"出现错误: {e}")

    return local_path


