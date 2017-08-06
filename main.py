import sys
import importlib
import main_script
import traceback
import operator

import feature_eng.utils as utils
import feature_eng.data_clean as data_clean
import feature_eng.feature_eng as feature_eng
import evaluator

data = None
training_data = None
last_config = None
i = 0
if __name__ == "__main__":
    while True:
        i += 1
        try:
            if not data:
                # Perserve the data in the memory, so do not need to read it again
                # between runs
                data = main_script.load_train_data()
            # Read config and reload modules
            config = main_script.load_config()
            develop_mode = config['develop_mode'] if 'develop_mode' in config else True
            if develop_mode:
                # if in develop_mode, reload all our modules so the changes can be
                # include dynamically.
                importlib.reload(utils)
                importlib.reload(data_clean)
                importlib.reload(feature_eng)
                importlib.reload(evaluator)
            # Determine if we need to re-process data
            force_process_data = config['force_process_data'] if 'force_process_data' in config else False
            if last_config is None or force_process_data:
                print('Force processing data')
                training_data = main_script.process_data(*data, config)
            else:
                steps = config['steps']
                training_preprocess = config['training_preprocess']
                predict = config['predict'] if 'predict' in config else False
                last_steps = last_config['steps']
                last_training_preprocess = last_config['training_preprocess']
                last_predict = last_config['predict'] if 'predict' in last_config else False
                if (not operator.eq(steps, last_steps)) or (not operator.eq(training_preprocess, last_training_preprocess)) or (not operator.eq(predict, last_predict)):
                    print('Data processing config changes', operator.eq(steps, last_steps), operator.eq(training_preprocess, last_training_preprocess), operator.eq(predict, last_predict))
                    training_data = main_script.process_data(*data, config)
            main_script.train(*training_data, config)
            last_config = config
        except BaseException as e:
            # Catch and print all exception, but prevent the program from exiting
            traceback.print_exc()
        finally:
            # Prevent somehow lose control of this loop
            if i == 100:
                break
            input("\nPress Enter to continue, CTRL-C to exit...\n")
            # Re load the change in main script
            importlib.reload(main_script)
