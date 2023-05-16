import os
import timeout_decorator
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import tempfile

from weights import WeightsDownloadCache


class TestWeightsDownloadCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.weights = WeightsDownloadCache(min_disk_space=1, base_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @timeout_decorator.timeout(1)
    @patch('shutil.disk_usage', return_value=MagicMock(free=0))
    @patch('os.makedirs')
    def test_init_creates_base_dir_if_not_exists(self, makedirs_mock, _):
        WeightsDownloadCache(min_disk_space=1, base_dir="/src/weights")
        makedirs_mock.assert_called_once_with("/src/weights")

    @timeout_decorator.timeout(1)
    @patch('shutil.disk_usage')
    @patch.object(WeightsDownloadCache, '_remove_least_recent')
    @patch('subprocess.check_output', return_value=b'Success')
    def test_download_weights_removes_least_recent_if_no_space(self, _, remove_mock, disk_usage_mock):
        # Initially, there's not enough space.
        disk_usage_mock.return_value = MagicMock(free=1)
        weights_url = "some_weights_orig"
        self.weights.ensure(weights_url)

        # After _remove_least_recent is called, there's enough space.
        def remove_side_effect():
            disk_usage_mock.return_value = MagicMock(free=2)
        remove_mock.side_effect = remove_side_effect

        weights_url = "some_weights"
        self.weights.ensure(weights_url)
        remove_mock.assert_called_once()

    @timeout_decorator.timeout(1)
    @patch('shutil.disk_usage', return_value=MagicMock(free=2))
    @patch('subprocess.check_output', return_value=b'Success')
    def test_download_weights_does_not_remove_if_enough_space(self, _, __):
        remove_mock = MagicMock()
        self.weights._remove_least_recent = remove_mock

        weights_url = "some_weights"
        self.weights.download_weights(weights_url, self.temp_dir.name)
        remove_mock.assert_not_called()

    @timeout_decorator.timeout(1)
    @patch.object(WeightsDownloadCache, 'download_weights')
    def test_ensure_downloads_weights_if_not_in_cache(self, download_mock):
        weights_url = "some_weights"
        path = self.weights.ensure(weights_url)
        download_mock.assert_called_once_with(weights_url, path)

    @timeout_decorator.timeout(1)
    @patch.object(WeightsDownloadCache, 'download_weights')
    def test_ensure_does_not_download_weights_if_in_cache(self, download_mock):
        weights_url = "some_weights"
        path = self.weights.weights_path(weights_url)
        self.weights.lru_paths.append(path)
        
        self.weights.ensure(weights_url)
        download_mock.assert_not_called()

    @timeout_decorator.timeout(1)
    @patch.object(WeightsDownloadCache, 'download_weights')
    def test_ensure_moves_existing_weights_to_end_of_cache(self, _):
        weights_url1 = "weights1"
        weights_url2 = "weights2"
        path1 = self.weights.weights_path(weights_url1)
        path2 = self.weights.weights_path(weights_url2)

        self.weights.lru_paths.append(path1)
        self.weights.lru_paths.append(path2)
        
        self.weights.ensure(weights_url1)
        
        self.assertEqual(self.weights.lru_paths[0], path2)
        self.assertEqual(self.weights.lru_paths[1], path1)


    @timeout_decorator.timeout(1)
    @patch('shutil.disk_usage')
    @patch('shutil.rmtree')
    @patch('subprocess.check_output', return_value=b'Success')
    def test_download_weights_removes_lru_when_not_enough_space(self, _, rmtree_mock, disk_usage_mock):
        # Initially, there's not enough space.
        disk_usage_mock.return_value = MagicMock(free=0)

        # After removing the least recent item, there's enough space.
        def remove_side_effect(*args, **kwargs):
            disk_usage_mock.return_value = MagicMock(free=2)
        rmtree_mock.side_effect = remove_side_effect

        # Add two items to the cache.
        weights_url1 = "weights1"
        weights_url2 = "weights2"
        path1 = self.weights.ensure(weights_url1)
        path2 = self.weights.ensure(weights_url2)

        # Download a new item, forcing the removal of the least recent item.
        weights_url3 = "weights3"
        self.weights.ensure(weights_url3)

        # Check if the least recent item (path1) was removed.
        rmtree_mock.assert_called_once_with(path1)

if __name__ == "__main__":
    unittest.main()
