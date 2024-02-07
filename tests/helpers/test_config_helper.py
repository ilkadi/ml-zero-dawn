import unittest
from ml_zero_dawn.helpers.config_helper import ConfigHelper

class TestConfigHelper(unittest.TestCase):
    def setUp(self):
        self.config_helper = ConfigHelper()

    def test_update_without_base(self):
        config_path = 'tests/helpers/resources/test_config.yaml'
        expected_config = {'key1': 'value1', 'key2': 'value2'}
        updated_config = self.config_helper.update_with_config(config_path)
        self.assertEqual(updated_config, expected_config)
        
    def test_update_with_config(self):
        base_config = {'key1': 'value1', 'key3': 'value3'}
        config_path = 'tests/helpers/resources/test_config.yaml'
        expected_config = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}

        updated_config = self.config_helper.update_with_config(config_path, base_config)

        self.assertEqual(updated_config, expected_config)

    def test_update_incorrect_path(self):
        config_path = 'tests/helpers/resources/test_config.yml'
        with self.assertRaises(ValueError):
            self.config_helper.update_with_config(config_path)

    def test_print_config(self):
        config = {'key1': 'value1', 'key2': 'value2'}
        expected_output = "Config:\n{'key1': 'value1', 'key2': 'value2'}\n"

        # Redirect stdout to capture the print output
        from io import StringIO
        import sys
        captured_output = StringIO()
        sys.stdout = captured_output

        self.config_helper.print_config(config)

        # Reset stdout
        sys.stdout = sys.__stdout__

        self.assertEqual(captured_output.getvalue(), expected_output)

if __name__ == '__main__':
    unittest.main()