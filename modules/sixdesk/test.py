import unittest

from da import davst


dbname = '/home/phermes/development/SixDeskDB/chrom-0.0-24.0.db'
d      = davst(dbname)

class davstTestCase(unittest.TestCase):

	def test_if_isinstance(self):
		self.assertIsInstance(d, davst)

	def test_fit_parameters(self):
		kk = d.fit_single_seed(1.99,2.01,0.01,1)
		results = (5.3630487451255702, 50.993998959495983, 1.99, 0.053386485008800007, 0.0058907190349086116, 1.174716817385435)
		for i in range(len(kk)):
			self.assertEqual(kk[i],results[i])

	


if __name__ == '__main__':
	unittest.main()