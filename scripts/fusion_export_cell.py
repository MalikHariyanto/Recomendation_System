# Sel ini disisipkan ke notebook; bukan CLI mandiri.
selection_train, selection_val = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
    random_state=2026).split(np.arange(len(QA)), groups=groupsA))
selection_fb, selection_gates = fit_gate_ensemble(FA, ZBa, ZEa, RELA, groupsA,
    VARIANTS[REF], selection_train, seed_split=2026)
selection_lambda = predict_ensemble(selection_fb, selection_gates, subset(FA, selection_val))
selection_scores = {
    'BGE-M3': float(ndcg_at(SA['BGE-M3'][selection_val], RELA[selection_val]).mean()),
    'E5': float(ndcg_at(SA['E5'][selection_val], RELA[selection_val]).mean()),
    REF: float(ndcg_at(fuse(selection_lambda, ZBa[selection_val], ZEa[selection_val]), RELA[selection_val]).mean()),
}
selected_method = max(selection_scores, key=selection_scores.get)
display(pd.Series(selection_scores, name='NDCG@10 validasi pemilihan').to_frame())
print('Calon serving:', selected_method, '| Layak deployment:', not MOCK)
print('Pemilihan ini berbasis validasi kelompok kategori, bukan oracle atau skor test.')

from pathlib import Path
import shutil
bundle = Path(OUT_DIR) / ('bundle_mock_DO_NOT_DEPLOY' if MOCK else 'bundle_streamlit')
bundle.mkdir(parents=True, exist_ok=True)
df[['destinasi','text','did']].to_json(bundle/'reviews.json', orient='records', force_ascii=False)
dest.to_json(bundle/'catalog.json', orient='records', force_ascii=False)
np.save(bundle/'bge.npy', RB, allow_pickle=False)
np.save(bundle/'e5.npy', RE, allow_pickle=False)
np.save(bundle/'offsets.npy', OFFSETS, allow_pickle=False)
np.savez(bundle/'preprocess.npz', pca_mean=fb_final.pca.mean_, pca_components=fb_final.pca.components_,
    scaler_mean=fb_final.scaler.mean_, scaler_scale=fb_final.scaler.scale_)
for i, gate in enumerate(gates_final):
    torch.save(gate.state_dict(), bundle/f'gate_{i}.pt')
manifest = {
    'schema':1, 'mock':MOCK, 'eligible_for_deployment':not MOCK,
    'selected_method':selected_method, 'selection_metric':'NDCG@10 grouped validation',
    'selection_scores':selection_scores, 'selection_train_query_ids':selection_train.tolist(),
    'selection_validation_query_ids':selection_val.tolist(),
    'dataset_sha256':DATASET_SHA256, 'reviews':int(N), 'destinations':int(D),
    'scope':'entire uploaded workbook; Jawa Timur; no Madura filter',
    'bge_model':CFG['bge_model'], 'bge_revision':CFG['bge_revision'],
    'e5_model':CFG['e5_model'], 'e5_revision':CFG['e5_revision'],
    'max_len':CFG['max_len'], 'topm':CFG['topm'], 'gate_hidden':CFG['gate_hidden'],
    'gate_dropout':CFG['gate_dropout'], 'ensemble_size':len(gates_final),
    'input_dim':int(fb_final.transform(FA).shape[1]),
    'query_context':['hujan','kemarau','none'],
    'evaluation_warning':'Silver metadata qrels; category groups can share relevant destinations; not human relevance judgments',
    'files':{},
}
for path in sorted(bundle.iterdir()):
    if path.name != 'manifest.json':
        manifest['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
(bundle/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
from wisata.fusion_serving import load_gate_ensemble, gate_features
check_ids = np.arange(min(5,len(QA)))
loaded_gates = load_gate_ensemble(bundle,manifest)
check_features = fb_final.transform(subset(FA,check_ids))
with torch.inference_mode():
    loaded_lambda = np.mean([torch.sigmoid(g(torch.from_numpy(check_features))).numpy().ravel()
                            for g in loaded_gates],axis=0)
np.testing.assert_allclose(loaded_lambda,predict_ensemble(fb_final,gates_final,subset(FA,check_ids)),atol=1e-6)
with np.load(bundle/'preprocess.npz',allow_pickle=False) as processor:
    check_row = gate_features(SA['BGE-M3'][:1],SA['E5'][:1],QAB[:1],QAE[:1],QA.season.iloc[0],QA.text.iloc[0],processor)
np.testing.assert_allclose(check_row,check_features[:1],atol=1e-5)
print('Pemeriksaan ekspor: lambda dan preprocessing bundle cocok dengan notebook.')
shutil.make_archive(str(bundle), 'zip', bundle)
print('Bundle:', bundle, '| ZIP:', str(bundle)+'.zip')
print('PDF setiap tahap:', Path(REPORT_DIR)/'hasil.pdf')
