import sys
import importlib
import main_script

data = None
i = 0
if __name__ == "__main__":
    while True:
        i += 1
        if not data:
            # Perserve the data in the memory, so do not need to read it again
            # between runs
            data = main_script.load_train_data()
        main_script.train(*data)
        # Prevent somehow lose control of this loop
        if i == 100:
            break
        input("Press Enter to continue, CTRL-C to exit...")
        # Re load the change in main script
        importlib.reload(main_script)
