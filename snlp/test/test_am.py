import unittest
from snlp.mwes.am import extract_ncs_from_sent, get_ams, get_counts


class TestAms(unittest.TestCase):

    def test_extract_nsc(self):
        self.assertRaises(TypeError, extract_ncs_from_sent, 5)
        self.assertRaises(TypeError, extract_ncs_from_sent, -2)
        self.assertRaises(TypeError, extract_ncs_from_sent, [1, 2, 3])
        self.assertRaises(TypeError, extract_ncs_from_sent, [1, 2, 'test'])
        self.assertRaises(ValueError, extract_ncs_from_sent, '')

if __name__ == '__main__':
    unittest.main()