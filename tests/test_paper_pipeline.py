import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from utils import EncoderLoss as losses
from utils import RecurrentClassifier as classifiers

spec = importlib.util.spec_from_file_location('snn_reference', Path(__file__).parent/'fixtures/snn_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class PaperPipelineTests(unittest.TestCase):
    def pipeline(self):
        path = Path(__file__).parents[1]/'utils/paper_pipeline.py'
        self.assertTrue(path.exists(), 'An explicit paper pipeline must restore the original SNN methods.')
        from utils import paper_pipeline
        return paper_pipeline

    def test_paper_profile_has_no_hidden_overrides(self):
        p = self.pipeline()
        for modality, batch in [('emg',32),('energy',8),('angle',16)]:
            c = p.paper_config(modality)
            self.assertEqual(c['batch_sz'], batch)
            self.assertEqual((c['encoder_epochs'],c['classifier_epochs'],c['num_steps']), (30,25,5))
            self.assertEqual((c['encoder_lr'],c['classifier_lr']), (.005,.00075))
            self.assertEqual((c['theta'],c['lif_beta'],c['drop_p']), (.99,.99,.5))
            self.assertEqual(c['task'], 'pain')
            self.assertEqual(c['classifier_layout'], 'snn_windows')
        c['batch_sz'] = 99
        self.assertEqual(p.paper_config('angle')['batch_sz'],16)

    def test_window_normalization_matches_original_without_mutating_input(self):
        p = self.pipeline()
        x = torch.tensor([[[2.,float('nan')],[4.,3.],[6.,7.]]])
        saved=x.clone()
        torch.testing.assert_close(p.normalize_window(x),reference.normalize_data2(x),rtol=0,atol=0)
        torch.testing.assert_close(x,saved,equal_nan=True)

    def test_stacked_encoder_matches_original_outputs_and_gradients(self):
        from utils.STAL import SpikeThresholdLearning
        torch.manual_seed(5)
        old=reference.ThresholdedFeatureExtractor(4,2,2,4,.5,False,4)
        new=SpikeThresholdLearning(4,2,2,4,4,.5)
        new.load_state_dict(old.state_dict())
        old.eval();new.eval()
        x=torch.rand(2,4,2)
        a=old(x);b=new(x)
        for u,v in zip(a,b):torch.testing.assert_close(u,v,rtol=0,atol=0)
        sum(t.sum() for t in a).backward();sum(t.sum() for t in b).backward()
        for (name,u),(other,v) in zip(old.named_parameters(),new.named_parameters()):
            self.assertEqual(name,other)
            torch.testing.assert_close(u.grad,v.grad,rtol=0,atol=0)

    def test_snn_loss_handles_all_zero_windows(self):
        self.assertIn('mode',__import__('inspect').signature(losses.EncoderLoss).parameters)
        x=torch.zeros(2,4)
        w=torch.zeros(2,8,requires_grad=True)
        loss=losses.EncoderLoss(mode='snn')(w,x,None,None)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(w.grad).all())

    def test_encoder_loss_matches_snn_value_and_gradients(self):
        self.assertIn('mode',__import__('inspect').signature(losses.EncoderLoss).parameters,
                      'Loss mode must distinguish original SNN arithmetic from the rewritten loss.')
        old=reference.EncoderLoss();new=losses.EncoderLoss(mode='snn')
        torch.manual_seed(7)
        x=torch.rand(2,8)+.1
        args=[torch.rand(2,16,requires_grad=True),torch.rand(2,8,requires_grad=True),torch.rand(2,8,requires_grad=True)]
        copied=[a.detach().clone().requires_grad_() for a in args]
        expected=old(args[0],x,args[1],args[2]);actual=new(copied[0],x,copied[1],copied[2])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        expected.backward();actual.backward()
        for a,b in zip(args,copied):
            if a.grad is None:self.assertIsNone(b.grad)
            else:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

    def test_recurrent_lif_matches_original_including_learnable_decay(self):
        cls=classifiers.FullRecordingRecurrentClassifier
        self.assertIn('learn_beta',__import__('inspect').signature(cls).parameters)
        torch.manual_seed(1957)
        old=reference.RecurrentClassifier3(16,lif_beta=.99,l1_sz=4,num_steps=5)
        new=cls(16,.99,4,5,learn_beta=True)
        new.load_state_dict(old.state_dict())
        x=torch.randn(2,16)
        a=old(x);b=new(x)
        for u,v in zip(a,b):torch.testing.assert_close(u,v,rtol=0,atol=0)
        a[1].sum().backward();b[1].sum().backward()
        for (name,u),(other,v) in zip(old.named_parameters(),new.named_parameters()):
            self.assertEqual(name,other)
            torch.testing.assert_close(u.grad,v.grad,rtol=0,atol=0)
        self.assertTrue(new.rlif1.beta.requires_grad)
        self.assertEqual(new(x)[0].shape[0],5)

    def test_original_sampling_draws_one_balanced_replacement_batch(self):
        p=self.pipeline()
        x=torch.arange(20.).reshape(10,2);y=torch.tensor([0]*8+[1]*2)
        loader=p.training_loader(x,y,4)
        self.assertEqual(loader.sampler.num_samples,8)
        self.assertEqual(loader.batch_size,8)
        self.assertEqual(len(loader),1)
        self.assertTrue(loader.sampler.replacement)
        torch.testing.assert_close(loader.sampler.weights,torch.tensor([1/8]*8+[1/2]*2,dtype=torch.double))

    def test_selected_checkpoint_does_not_change_with_live_parameters(self):
        p=self.pipeline();net=torch.nn.Linear(2,1)
        state=p.snapshot(net);expected=state['weight'].clone()
        with torch.no_grad():net.weight.add_(3)
        torch.testing.assert_close(state['weight'],expected,rtol=0,atol=0)

    def test_decision_uses_last_complete_window_and_density_uses_all_windows(self):
        p=self.pipeline()
        class Encoder(torch.nn.Module):
            def forward(self,x):return x.flatten(1),None,None
        class Classifier(torch.nn.Module):
            def forward(self,x):
                # Count spikes as two-class votes; one recurrent step suffices here.
                score=x.sum(1)
                spikes=torch.stack([1-score,score],dim=1).unsqueeze(0)
                return spikes,spikes
        # Three windows; final incomplete tail must be ignored.
        x=torch.tensor([[[0.],[0.],[1.],[1.],[0.],[1.],[99.]]])
        config=dict(window_size=2,stride=2,theta=.5)
        pred,density=p.predict_recording(Encoder(),Classifier(),x,config,torch.device('cpu'))
        self.assertEqual(pred.tolist(),[1])
        self.assertAlmostEqual(density,1/6)  # Constant windows normalize to zero in SNN.

    def test_complete_tiny_fold_saves_independent_models_splits_and_predictions(self):
        p=self.pipeline()
        torch.manual_seed(11);np.random.seed(11)
        x=torch.rand(12,24,2);y=torch.tensor([0]*8+[1]*4)
        c=p.paper_config('emg')
        c.update(window_size=4,stride=2,n_spikes_per_timestep=2,l1_sz=4,l2_sz=4,l1_cls=4,
                 batch_sz=2,encoder_epochs=2,classifier_epochs=2,num_steps=2)
        with tempfile.TemporaryDirectory() as tmp:
            result=p.train_fold(c,x,y,np.array([i for i in range(12) if i!=8]),np.array([8]),8,torch.device('cpu'),Path(tmp))
            self.assertEqual(result['test_labels'],[1])
            self.assertEqual(len(result['test_preds']),1)
            self.assertTrue(0<=result['sparsity']<=1)
            self.assertTrue((Path(tmp)/'encoder.pt').exists())
            self.assertTrue((Path(tmp)/'classifier.pt').exists())
            self.assertTrue((Path(tmp)/'details.json').exists())


class RunnerContractTests(unittest.TestCase):
    def test_resume_rejects_changed_configuration_and_data(self):
        import reproduce
        self.assertTrue(hasattr(reproduce, 'prepare_run'), 'Resume must validate a complete provenance record.')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            original={'config':{'batch_sz':32},'data_sha256':{'data_Cs.pickle':'abc'}}
            reproduce.prepare_run(path,original,resume=False)
            reproduce.prepare_run(path,original,resume=True)
            for changed in [{'config':{'batch_sz':4},'data_sha256':{'data_Cs.pickle':'abc'}},
                            {'config':{'batch_sz':32},'data_sha256':{'data_Cs.pickle':'different'}}]:
                with self.assertRaises(ValueError):reproduce.prepare_run(path,changed,resume=True)
            with self.assertRaises(ValueError):reproduce.prepare_run(path,original,resume=False)
