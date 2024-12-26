from pynput.keyboard import Controller, Key
import time

def emit_keypress_from_string(input_string, delay=0.1):
    """
    Simulates keypress events for each character in the input string.

    Args:
        input_string (str): The string to emit as keypress events.
        delay (float): Delay between each keypress in seconds.
    """
    keyboard = Controller()

    for char in input_string:
        if char == '\n':  # Handle newline
            keyboard.press(Key.enter)
            keyboard.release(Key.enter)
        elif char == '\t':  # Handle tab
            keyboard.press(Key.tab)
            keyboard.release(Key.tab)
        elif char == ' ':  # Handle space
            keyboard.press(Key.space)
            keyboard.release(Key.space)
        else:
            keyboard.press(char)
            keyboard.release(char)
        time.sleep(delay)  # Add delay between keypresses

if __name__ == "__main__":
    # Example usage
    input_string = "Hello, World!\nThis is a test string.\tEnjoy!"
    emit_keypress_from_string(input_string, delay=0.2)