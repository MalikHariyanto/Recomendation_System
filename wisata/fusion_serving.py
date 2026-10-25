"""Serving bundle hasil notebook Malik; mock ditolak untuk rekomendasi pengguna."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def zscore(scores):
    scores = np.asarray(scores, dtype=np.float32)
    return (scores-scores.mean(1,keepdims=True))/(scores.std(1,keepdims=True)+1e-9)


def qpp(scores):
    ordered = -np.sort(-scores,axis=1)
    z = -np.sort(-zscore(scores),axis=1)
    probability = np.exp(z[:,:10]-z[:,:1]); probability /= probability.sum(1,keepdims=True)
    return np.c_[ordered[:,0], ordered[:,0]-ordered[:,min(1,ordered.shape[1]-1)],
        ordered[:,:10].mean(1),ordered[:,:10].std(1),scores.std(1),z[:,0],
        z[:,0]-z[:,min(9,z.shape[1]-1)],-(probability*np.log(probability+1e-12)).sum(1)]


def gate_features(sb,se,qb,qe,season,text,preprocessor):
    a,b=np.argsort(-sb,axis=1,kind='stable')[:,:10],np.argsort(-se,axis=1,kind='stable')[:,:10]
    jac=np.array([len(set(x)&set(y))/len(set(x)|set(y)) for x,y in zip(a,b)])
    ra,rb=rankdata(sb,axis=1),rankdata(se,axis=1)
    ra-=ra.mean(1,keepdims=True);rb-=rb.mean(1,keepdims=True)
    rho=(ra*rb).sum(1)/np.maximum(np.sqrt((ra**2).sum(1)*(rb**2).sum(1)),1e-12)
    numeric=np.c_[qpp(sb),qpp(se),jac,rho,np.log1p([len(text.split())])].astype(np.float32)
    emb=np.hstack([qb,qe]).astype(np.float32)
    reduced=(emb-preprocessor['pca_mean'])@preprocessor['pca_components'].T
    values=np.hstack([numeric,reduced]).astype(np.float32)
    values=((values-preprocessor['scaler_mean'])/preprocessor['scaler_scale']).astype(np.float32)
    ctx=np.zeros((len(sb),3),np.float32);ctx[:,['hujan','kemarau','none'].index(season)]=1
    return np.hstack([values,ctx]).astype(np.float32)


def load_gate_ensemble(directory,meta):
    import torch
    from torch import nn
    result=[]
    for i in range(meta['ensemble_size']):
        network=nn.Sequential(nn.Linear(meta['input_dim'],meta['gate_hidden']),nn.GELU(),
            nn.Dropout(meta['gate_dropout']),nn.Linear(meta['gate_hidden'],1))
        raw=torch.load(directory/f'gate_{i}.pt',map_location='cpu',weights_only=True)
        network.load_state_dict({key.removeprefix('net.'):value for key,value in raw.items()})
        result.append(network.eval())
    return result


class FusionService:
    def __init__(self,directory:Path):
        self.directory=directory
        self.meta=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if self.meta['mock'] or not self.meta['eligible_for_deployment']:
            raise ValueError('Bundle mock hanya untuk pengujian, tidak boleh dipakai rekomendasi penelitian.')
        for name,digest in self.meta['files'].items():
            if Path(name).name != name or hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Checksum bundle tidak cocok: '+name)
        self.reviews=pd.read_json(directory/'reviews.json',orient='records')
        self.catalog=pd.read_json(directory/'catalog.json',orient='records')
        self.bge=np.load(directory/'bge.npy',allow_pickle=False).astype(np.float32)
        self.e5=np.load(directory/'e5.npy',allow_pickle=False).astype(np.float32)
        self.offsets=np.load(directory/'offsets.npy',allow_pickle=False)
        if (len(self.bge)!=len(self.reviews) or len(self.e5)!=len(self.reviews)
                or len(self.offsets)!=len(self.catalog)+1 or self.offsets[-1]!=len(self.reviews)):
            raise ValueError('Urutan/dimensi korpus bundle tidak sesuai.')
        self.encoders={}
        self.gates=None

    def encode(self,kind,text):
        if kind not in self.encoders:
            from sentence_transformers import SentenceTransformer
            self.encoders[kind]=SentenceTransformer(self.meta[kind+'_model'],revision=self.meta[kind+'_revision'],
                device='cpu',model_kwargs={'use_safetensors':kind=='e5'})
            self.encoders[kind].max_seq_length=self.meta['max_len']
        prefix='query: ' if kind=='e5' else ''
        return self.encoders[kind].encode([prefix+text],normalize_embeddings=True,
            convert_to_numpy=True).astype(np.float16).astype(np.float32)

    def destination_scores(self,query,corpus):
        similarity=query@corpus.T
        result=[]
        for start,stop in zip(self.offsets[:-1],self.offsets[1:]):
            segment=similarity[:,start:stop]
            count=min(self.meta['topm'],segment.shape[1])
            result.append(np.sort(segment,axis=1)[:,-count:].mean(1))
        return np.stack(result,axis=1)

    def predict_lambda(self,sb,se,qb,qe,season,text):
        import torch
        if self.gates is None:
            self.gates=load_gate_ensemble(self.directory,self.meta)
        with np.load(self.directory/'preprocess.npz',allow_pickle=False) as processor:
            features=gate_features(sb,se,qb,qe,season,text,processor)
        with torch.inference_mode():
            values=[torch.sigmoid(gate(torch.from_numpy(features))).item() for gate in self.gates]
        return float(np.mean(values))

    def search(self,text,season='none',k=10,filter_season=True):
        if not text.strip() or season not in ['hujan','kemarau','none'] or k<1:
            raise ValueError('Isi query dan pilih konteks yang valid.')
        method=self.meta['selected_method'];lam=None
        if method=='BGE-M3':
            qb=self.encode('bge',text);scores=self.destination_scores(qb,self.bge)[0];review=(qb@self.bge.T)[0]
        elif method=='E5':
            qe=self.encode('e5',text);scores=self.destination_scores(qe,self.e5)[0];review=(qe@self.e5.T)[0]
        elif method=='Adaptive Fusion (usulan)':
            qb,qe=self.encode('bge',text),self.encode('e5',text)
            sb,se=self.destination_scores(qb,self.bge),self.destination_scores(qe,self.e5)
            lam=self.predict_lambda(sb,se,qb,qe,season,text)
            scores=(lam*zscore(sb)+(1-lam)*zscore(se))[0]
            review=(lam*(qb@self.bge.T)+(1-lam)*(qe@self.e5.T))[0]
        else:raise ValueError('Metode deployment tidak didukung.')
        valid=np.isfinite(scores)
        if filter_season and season!='none':
            valid &= self.catalog['cocok_hujan' if season=='hujan' else 'cocok_kemarau'].to_numpy(bool)
        ids=np.flatnonzero(valid);ids=ids[np.argsort(-scores[ids],kind='stable')[:k]]
        result=self.catalog.iloc[ids][['destinasi','wisata','musim','rating_mean']].copy()
        result['skor']=scores[ids]
        result['ulasan_bukti']=[self.reviews['text'].iloc[int(self.offsets[i])+int(np.argmax(review[self.offsets[i]:self.offsets[i+1]]))] for i in ids]
        return result.reset_index(drop=True),{'method':method,'lambda_bge':lam}
