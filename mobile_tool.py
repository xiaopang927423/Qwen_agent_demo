from typing import Union, Tuple, List

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool("mobile_use")
class MobileUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "click",
                    "long_press",
                    "swipe",
                    "type",
                    "system_button",
                    "open",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
                "type": "array",
            },
            "coordinate2": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=key`, `action=type`, and `action=open`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                "type": "number",
            },
            "button": {
                "description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`",
                "enum": [
                    "Back",
                    "Home",
                    "Menu",
                    "Enter",
                ],
                "type": "string",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None,device=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        self.device = device
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action == "key":
            return self._key(params["text"])
        elif action == "click":
            return self._click(
                coordinate=params["coordinate"]
            )
        elif action == "long_press":
            return self._long_press(
                coordinate=params["coordinate"], time=params["time"]
            )
        elif action == "swipe":
            return self._swipe(
                coordinate=params["coordinate"], coordinate2=params["coordinate2"]
            )
        elif action == "type":
            return self._type(params["text"])
        elif action == "system_button":
            return self._system_button(params["button"])
        elif action == "open":
            return self._open(params["text"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Unknown action: {action}")

    def _key(self, text: str):
            command = "input keyevent {}".format(text)
            self.device.shell(command)



    def _click(self, coordinate: Tuple[int, int]):
        x, y = coordinate
        command = f"input tap {x} {y}"
        self.device.shell(command)

    def _long_press(self, coordinate: Tuple[int, int], time: int):
        x, y = coordinate
        command = f"input swipe {x} {y} {x} {y} {int(time * 1000)}"
        self.device.shell(command)



    def _swipe(self, coordinate: Tuple[int, int], coordinate2: Tuple[int, int]):
        x1, y1 = coordinate
        x2, y2 = coordinate2
        self.device.shell(f"input swipe {x1} {y1} {x2} {y2} 1000")

    def _type(self, text: str):
        self.device.shell(f"am broadcast -a ADB_INPUT_TEXT --es msg {text}")


    def _system_button(self, button: str):
        button = button.upper()
        self.device.shell(f"input keyevent {button}")

    def _open(self, text: str):
        self.device.shell(f"am start -a android.intent.action.VIEW -d {text}")

    def _wait(self, time: int):
        self.device.shell(f"sleep {time}")

    def _terminate(self, status: str):
        pass