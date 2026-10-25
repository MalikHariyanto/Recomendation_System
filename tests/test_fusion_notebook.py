import ast
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

nbformat=pytest.importorskip('nbformat')
ROOT=Path(__file__).resolve().parents[1]


def notebook_defs(*names, **extra):
    notebook=nbformat.read(ROOT/'notebooks/Malik_AdaptiveFusion_BGE_E5_Colab.ipynb',as_version=4)
    nodes=[]
    for cell in notebook.cells:
        if cell.cell_type=='code':
            tree=ast.parse(cell.source)
            top=[]
            for statement in tree.body:
                top.extend(statement.body if isinstance(statement,ast.With) else [statement])
            nodes += [node for node in top if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    namespace={'np':np,'pd':pd,'stats':stats,'PCA':PCA,'StandardScaler':StandardScaler,
               'Path':Path,'hashlib':hashlib,'json':json,**extra}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<notebook-definitions>','exec'),namespace)
    return namespace


def test_zero_relevance_and_small_catalog_metrics_are_finite():
    ns=notebook_defs('zscore','topk_idx','ndcg_at','evaluate',CAT_CODE=np.array([0,1,1]),NOVELTY=np.ones(3))
    scores=np.array([[3.,2.,1.]])
    output=ns['evaluate'](scores,np.zeros((1,3)),k=10)
    assert output['MAP@10'][0]==0
    assert all(np.isfinite(value).all() for value in output.values())


def test_qpp_ties_and_serving_preprocessor_match_notebook():
    from wisata.fusion_serving import gate_features
    ns=notebook_defs('zscore','topk_idx','qpp_block','agreement','build_feats','subset','GateFeatures',
                     CFG={'pca_dim':2})
    ns['CTX_IDX']={'hujan':0,'kemarau':1,'none':2}
    rng=np.random.default_rng(42)
    sb,se=rng.random((6,4)),rng.random((6,4))
    qb,qe=rng.random((6,3)),rng.random((6,3))
    sb[0]=0;se[0]=0
    text=['wisata keluarga']*6
    features=ns['build_feats'](sb,se,qb,qe,['hujan']*6,text)
    assert np.isfinite(features['qpp']).all()
    fb=ns['GateFeatures'](['qpp','emb','ctx']).fit(features)
    processor={'pca_mean':fb.pca.mean_,'pca_components':fb.pca.components_,
               'scaler_mean':fb.scaler.mean_,'scaler_scale':fb.scaler.scale_}
    expected=fb.transform(ns['subset'](features,np.array([0])))
    actual=gate_features(sb[:1],se[:1],qb[:1],qe[:1],'hujan',text[0],processor)
    np.testing.assert_allclose(actual,expected,atol=1e-5)


def test_embedding_cache_detects_middle_change_and_corruption(tmp_path):
    import torch
    class Encoder:
        calls=0
        def encode(self,texts,**kwargs):
            self.calls+=1
            return np.ones((len(texts),3),np.float32)
    ns=notebook_defs('encode',MOCK=True,CACHE=str(tmp_path),torch=torch,os=__import__('os'),
        CFG={'batch_size':8,'max_len':256,'bge_model':'BGE','bge_revision':'1','e5_model':'E5','e5_revision':'2','seed':42})
    model=Encoder();texts=['awal','tengah','akhir']
    ns['encode'](model,texts,name='test');ns['encode'](model,texts,name='test')
    assert model.calls==1
    ns['encode'](model,['awal','berubah','akhir'],name='test')
    assert model.calls==2
    for path in tmp_path.glob('*.npy'):path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Checksum'):
        ns['encode'](model,texts,name='test')


def test_pdf_is_saved_on_failed_stage_and_marked_mock(tmp_path):
    pytest.importorskip('reportlab')
    from pypdf import PdfReader
    from wisata.pdf_results import ResultRecorder
    recorder=ResultRecorder(str(tmp_path),mock=True)
    with pytest.raises(ValueError):
        with recorder.stage('uji kegagalan'):
            recorder.display(pd.DataFrame({'metode':['fixture'],'NDCG':[.5]}))
            raise ValueError('test failure')
    text=''.join(page.extract_text() for page in PdfReader(tmp_path/'hasil.pdf').pages)
    assert 'MOCK' in text and 'failed' in text and 'test failure' in text
    assert json.loads((tmp_path/'catatan_tahap.json').read_text())['stages'][0]['status']=='failed'


def test_mock_bundle_is_never_served(tmp_path):
    from wisata.fusion_serving import FusionService
    (tmp_path/'manifest.json').write_text(json.dumps({'mock':True,'eligible_for_deployment':False}))
    with pytest.raises(ValueError,match='mock'):
        FusionService(tmp_path)
