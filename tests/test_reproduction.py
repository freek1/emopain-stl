import importlib.util
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
import torch

from utils import RecurrentClassifier as classifiers
from utils.load_data import load_data_emopain


class ReproductionTests(unittest.TestCase):
    def test_full_recording_classifier_matches_archived_dynamics_and_gradients(self):
        cls = getattr(classifiers, 'FullRecordingRecurrentClassifier', None)
        self.assertIsNotNone(cls, 'The reproduction must support the historical full-recording classifier.')
        spec = importlib.util.spec_from_file_location('archived_classifier', Path(__file__).parent/'fixtures'/'reference_classifier.py')
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        torch.manual_seed(1957)
        old = reference.RecurrentClassifier(60, .99, 5, 0, 2, 10)
        new = cls(input_size=60, lif_beta=.99, hidden_size=5, num_steps=10)
        new.load_state_dict(old.state_dict())
        x = torch.randn(2, 60)
        old_out = old(x)
        new_out = new(x)
        for a, b in zip(old_out, new_out):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        old_out[1].sum().backward()
        new_out[1].sum().backward()
        for (name, a), (other, b) in zip(old.named_parameters(), new.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        changed=x.clone(); changed[:, 12:] += 10
        self.assertFalse(torch.equal(new(x)[1], new(changed)[1]), 'Late recording values must affect the classifier.')

    def test_pain_loader_uses_requested_directory_and_original_labels(self):
        import inspect
        self.assertIn('data_dir', inspect.signature(load_data_emopain).parameters, 'Data location must be explicit for the reproduction.')
        self.assertIn('task', inspect.signature(load_data_emopain).parameters, 'Pain and protective labels must be selected explicitly.')
        with tempfile.TemporaryDirectory() as tmp:
            for group, label in [('C',0),('P',1)]:
                record={f'sEMG_probe_{i}':np.array([1.,2.,3.,4.]) for i in range(1,5)}
                record['protective_label']=label
                with open(Path(tmp)/f'data_{group}s.pickle','wb') as f:pickle.dump([record],f)
            x,y=load_data_emopain('emg',data_dir=tmp,task='pain')
            self.assertEqual(x.shape,(2,4,4))
            self.assertEqual(y.tolist(),[0,1])
            torch.testing.assert_close(x[0,:,0],torch.tensor([0.,1/3,2/3,1.]))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
