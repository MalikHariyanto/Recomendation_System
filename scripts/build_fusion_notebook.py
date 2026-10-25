"""Bangun notebook Malik dari notebook fusion yang diberikan pengguna."""
import ast
from pathlib import Path
import textwrap

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Patch sumber tidak cocok: " + old[:90])
    return source.replace(old, new, 1)


BOOTSTRAP = '''import os, sys, subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid
RUN_MODE = os.environ.get('FUSION_RUN_MODE', 'research')  # Colab: research atau smoke
MOCK = RUN_MODE == 'smoke'
os.environ['MOCK_ENCODER'] = '1' if MOCK else '0'
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    repo = Path('/content/Recomendation_System_fusion')
    if not repo.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'feat/malik-adaptive-fusion-colab',
                        'https://github.com/MalikHariyanto/Recomendation_System.git', str(repo)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(repo/'requirements-fusion-colab.txt')], check=True)
    sys.path.insert(0, str(repo))
    WORK_DIR = '/content/drive/MyDrive/Rekomendasi_Wisata'
else:
    repo = Path(os.environ.get('FUSION_REPO', '.')).resolve()
    sys.path.insert(0, str(repo))
    WORK_DIR = os.environ.get('WORK_DIR', '.')
DATA_NAME = 'koreksi_dataset.xlsx'
os.makedirs(WORK_DIR, exist_ok=True)
os.environ.setdefault('HF_HOME', os.path.join(WORK_DIR, 'hf_cache'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from wisata.pdf_results import ResultRecorder
RUN_ID = os.environ.get('FUSION_RUN_ID') or (datetime.now(ZoneInfo('Asia/Bangkok')).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
REPORT_DIR = os.path.join(WORK_DIR, 'reports_mock' if MOCK else 'reports_research', RUN_ID)
recorder = ResultRecorder(REPORT_DIR, MOCK)
display = recorder.display
print('Malik | Adaptive Fusion | Mode:', RUN_MODE, '| Folder:', WORK_DIR)
print('Repository SHA:', subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip())
'''


ENCODING = '''class MockEncoder:
    """Encoder fixture hanya untuk uji alur; bukan pengganti model penelitian."""
    def __init__(self, corpus, analyzer, ngram, dim, seed):
        self.vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, max_features=30000, sublinear_tf=True)
        matrix = self.vec.fit_transform(corpus)
        dim = min(dim, matrix.shape[0]-1, matrix.shape[1]-1)
        self.svd = TruncatedSVD(dim, random_state=seed).fit(matrix)
    def encode(self, texts, **kw):
        texts = [re.sub(r'^(query: |passage: )', '', text) for text in texts]
        vectors = self.svd.transform(self.vec.transform(texts))
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

if MOCK:
    enc_bge = MockEncoder(df['text'], 'char_wb', (3, 5), 32, 1)
    enc_e5 = MockEncoder(df['text'], 'word', (1, 2), 32, 2)
else:
    from sentence_transformers import SentenceTransformer
    enc_bge = SentenceTransformer(CFG['bge_model'], revision=CFG['bge_revision'], device=DEVICE,
                                  model_kwargs={'use_safetensors':False})
    enc_e5 = SentenceTransformer(CFG['e5_model'], revision=CFG['e5_revision'], device=DEVICE,
                                 model_kwargs={'use_safetensors':True})
    for model in (enc_bge, enc_e5):
        model.max_seq_length = CFG['max_len']
        if DEVICE == 'cuda': model.half()

def encode(model, texts, prefix='', name=None):
    texts = [prefix + str(text) for text in texts]
    if not texts: raise ValueError('Daftar teks kosong')
    identity = {'name':name, 'mock':MOCK, 'texts':texts, 'prefix':prefix, 'max_len':CFG['max_len'],
                'bge':CFG['bge_model'], 'bge_revision':CFG['bge_revision'],
                'e5':CFG['e5_model'], 'e5_revision':CFG['e5_revision'], 'seed':CFG['seed'],
                'dtype':'float16-storage', 'normalized':True, 'preprocess':'ftfy-whitespace-min3-v1'}
    digest = hashlib.sha256(json.dumps(identity,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    path = os.path.join(CACHE, f'{name}_{digest}.npy') if name else None
    if path and os.path.exists(path) and os.path.exists(path+'.json'):
        meta = json.loads(Path(path+'.json').read_text(encoding='utf-8'))
        if meta['sha256'] != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError('Checksum cache embedding tidak cocok')
        result = np.load(path, allow_pickle=False)
        if len(result) != len(texts) or not np.isfinite(result).all(): raise ValueError('Cache tidak valid')
        return result
    try:
        result = model.encode(texts, batch_size=CFG['batch_size'], normalize_embeddings=True,
                              show_progress_bar=bool(name), convert_to_numpy=True).astype(np.float16)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result = model.encode(texts, batch_size=max(1,CFG['batch_size']//4), normalize_embeddings=True,
                              show_progress_bar=bool(name), convert_to_numpy=True).astype(np.float16)
    if not np.isfinite(result).all(): raise ValueError('Embedding tidak finite')
    if path:
        tmp = path+'.tmp'
        with open(tmp,'wb') as file: np.save(file,result,allow_pickle=False)
        os.replace(tmp,path)
        Path(path+'.json').write_text(json.dumps({'identity_sha256':digest,'shape':list(result.shape),
            'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}),encoding='utf-8')
    return result

t0 = time.time()
RB = encode(enc_bge, df['text'].tolist(), '', 'bge_corpus')
RE = encode(enc_e5, df['text'].tolist(), CFG['e5_doc_prefix'], 'e5_corpus')
QAB = encode(enc_bge, QA['text'].tolist(), '', 'bge_qA')
QAE = encode(enc_e5, QA['text'].tolist(), CFG['e5_query_prefix'], 'e5_qA')
QBB = encode(enc_bge, QB_['text'].tolist(), '', 'bge_qB')
QBE = encode(enc_e5, QB_['text'].tolist(), CFG['e5_query_prefix'], 'e5_qB')
print(f'Encoding selesai {time.time()-t0:.1f}s | BGE {RB.shape}, E5 {RE.shape}')
'''


def build():
    nb = nbformat.read(ROOT/'notebooks/source/3_AdaptiveFusion_sumber_pengguna.ipynb', as_version=4)
    cells = nb.cells
    cells[0].source = '''# Malik — Adaptive Fusion BGE-M3 + E5 (Colab)

Tugas kelompok Adaptive Fusion (PJ Faza). Notebook ini memperbaiki notebook pengguna,
memakai BGE sebagai baseline dan E5 sebagai komponen/pembanding. **Metode tugas ini
adalah fusion embedding, bukan LDA/FastText/BiLSTM pada jalur proposal sebelumnya.**

Pilih **Runtime → Change runtime type → T4 GPU**, simpan workbook di
`MyDrive/Rekomendasi_Wisata/koreksi_dataset.xlsx`, lalu Run all. Cakupan default
adalah seluruh workbook Jawa Timur, bukan filter Madura. Folder cache/output/PDF
disimpan di Drive. Hasil real dan mock dipisahkan. PDF diperbarui setiap tahap;
CSV menyimpan seluruh nilai. PDF pada tahap gagal tetap memuat error.

Baseline BGE tidak dilatih ulang. Embedding dipakai ulang hanya bila manifest dan
checksum cocok; cache lama tanpa identitas penuh tidak diasumsikan valid.
`FUSION_RUN_MODE=smoke` hanya untuk uji kode, bukan hasil penelitian.
Paper/riwayat koreksi data yang disebut notebook sumber belum semuanya diverifikasi.
'''
    cells[1].source = BOOTSTRAP
    cells[2].source = cells[2].source.replace("try:\n    from ftfy import fix_text          # memperbaiki mojibake (mis. \"ðŸ˜­\" → \"😭\")\nexcept ImportError:\n    fix_text = lambda s: s", "from ftfy import fix_text")
    cells[2].source = cells[2].source.replace("warnings.filterwarnings('ignore')", "# Warning penting tidak disembunyikan.\nif MOCK: torch.set_num_threads(2)")
    cells[2].source += '''\n_original_show = plt.show
def _show(*args, **kwargs):
    recorder.capture_figures()
    _original_show(*args, **kwargs)
    plt.close('all')
plt.show = _show
'''
    cells[3].source = cells[3].source.replace("seed            = 42,", "seed            = 42,\n    bge_revision = '5617a9f61b028005a4858fdac845db406aefb181',\n    e5_revision = '3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3',")
    cells[3].source = cells[3].source.replace("CFG.update(n_known_item=500, gate_epochs=150, gate_seeds=[0], mock_cap=40)",
        "CFG.update(n_known_item=200, gate_epochs=20, gate_patience=5, gate_seeds=[0], mock_cap=8, n_folds=3, n_boot=100)")
    cells[3].source = cells[3].source.replace("OUT_DIR = os.environ.get('OUT_DIR', os.path.join(WORK_DIR, 'output_fusion'))", "OUT_DIR = os.environ.get('OUT_DIR', os.path.join(WORK_DIR, 'output_fusion_mock' if MOCK else 'output_fusion_research', RUN_ID))")
    cells[3].source = cells[3].source.replace("if IS_COLAB and DEVICE != 'cuda':", "if not MOCK and DEVICE != 'cuda':\n    raise RuntimeError('Run penelitian memerlukan GPU. Aktifkan T4 di Colab; gunakan smoke hanya untuk tes kode.')\nif IS_COLAB and DEVICE != 'cuda':")
    cells[3].source += "\nDATASET_SHA256 = hashlib.sha256(Path(DATA_PATH).read_bytes()).hexdigest()\nprint('Dataset SHA256:', DATASET_SHA256)\n"
    cells[8].source = cells[8].source.replace("asp_strong = (asp_frac >= asp_frac.quantile(0.6))", "asp_strong = ((asp_frac > 0) & (asp_frac >= asp_frac.quantile(0.6)))")
    cells[11].source = ENCODING
    cells[12].source = cells[12].source.replace('dan tidak bias ke destinasi yang ulasannya banyak', 'tetap dapat dipengaruhi jumlah ulasan; bebas bias tidak diasumsikan')
    metrics = cells[15].source
    metrics = metrics.replace("disc = 1 / np.log2", "k = min(k, S.shape[1])\n    disc = 1 / np.log2", 1)
    metrics = metrics.replace("top = topk_idx(S, k)\n", "k = min(k, S.shape[1])\n    top = topk_idx(S, k)\n", 1)
    metrics = metrics.replace("np.minimum(nrel, k),", "np.maximum(np.minimum(nrel, k), 1),")
    metrics = metrics.replace("1 - same_pairs / (k * (k - 1))", "(1 - same_pairs / (k * (k - 1))) if k > 1 else np.zeros(len(S))")
    metrics = metrics.replace("rb = np.argsort(np.argsort(-SB, 1), 1) + 1", "rb = stats.rankdata(-SB, axis=1, method='average')")
    metrics = metrics.replace("re_ = np.argsort(np.argsort(-SE, 1), 1) + 1", "re_ = stats.rankdata(-SE, axis=1, method='average')")
    cells[15].source = metrics
    gate = cells[19].source
    gate = gate.replace("z[:, 9]", "z[:, min(9, z.shape[1]-1)]")
    gate = gate.replace("srt[:, 1]", "srt[:, min(1, srt.shape[1]-1)]")
    gate = gate.replace("ra = np.argsort(np.argsort(SB, 1), 1).astype(float); rb = np.argsort(np.argsort(SE, 1), 1).astype(float)", "ra = stats.rankdata(SB, axis=1).astype(float); rb = stats.rankdata(SE, axis=1).astype(float)")
    gate = gate.replace("np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))", "np.maximum(np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1)), 1e-12)")
    gate = gate.replace("PCA(min(CFG['pca_dim'], len(Fd['qemb']) - 1)", "PCA(min(CFG['pca_dim'], len(Fd['qemb']) - 1, Fd['qemb'].shape[1])")
    gate = gate.replace("lam_fix[te] = best_fixed_lambda", "assert not set(groups[tr]) & set(groups[te]), 'Group leakage'\n        lam_fix[te] = best_fixed_lambda")
    cells[19].source = gate
    cells[18].source += '\nID kategori query dipisahkan; destinasi relevan masih bisa beririsan karena super-kategori. Ini bukan holdout destinasi baru.\n'
    stats_cell = cells[22].source
    stats_cell = stats_cell.replace("def significance(perq, ref, metric, exclude=()):", "def significance(perq, ref, metric, exclude=(), groups=None):")
    stats_cell = stats_cell.replace("b_ = np.nan_to_num(perq[m][metric]); d = a - b_", "b_ = np.nan_to_num(perq[m][metric]); d = a - b_\n        if groups is not None: d = pd.Series(d).groupby(np.asarray(groups)).mean().values")
    stats_cell = stats_cell.replace("stats.ttest_rel(a, b_).pvalue if d.std() > 0 else 1.0", "stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else np.nan")
    stats_cell = stats_cell.replace("stats.wilcoxon(a, b_, zero_method='zsplit').pvalue if d.std() > 0 else 1.0", "stats.wilcoxon(d, zero_method='zsplit').pvalue if np.any(np.abs(d)>1e-12) else 1.0")
    stats_cell = stats_cell.replace("def boot_ci(x, n=CFG['n_boot'], seed=0):", "def boot_ci(x, n=CFG['n_boot'], seed=0):")
    stats_cell = stats_cell.replace("means = x[r.integers(0, len(x), (n, len(x)))].mean(1)", "groups = QA['cat'].values\n    tab = pd.DataFrame({'x':x,'group':groups}).groupby('group').x.agg(['sum','size'])\n    sampled = r.integers(0,len(tab),(n,len(tab)))\n    means = tab['sum'].values[sampled].sum(1) / tab['size'].values[sampled].sum(1)")
    stats_cell = stats_cell.replace("exclude=('Oracle λ per query (batas atas)',))", "exclude=('Oracle λ per query (batas atas)',), groups=QA['cat'].values)")
    cells[22].source = stats_cell
    cells[28].source = cells[28].source.replace("exclude=('Oracle λ per query (batas atas)',))", "exclude=('Oracle λ per query (batas atas)',), groups=QB_['did'].values)")
    cells[32].source = cells[32].source.replace("significance(perqD, REF, 'NDCG@10')", "significance(perqD, REF, 'NDCG@10', groups=QD['cfg'].values)")
    # Jangan mengisi ranking dengan kandidat -inf setelah filter.
    cells[34].source = cells[34].source.replace("top = np.argsort(-s, kind='stable')[:k]", "valid = np.flatnonzero(np.isfinite(s))\n    top = valid[np.argsort(-s[valid], kind='stable')[:k]]")
    # Pemilihan calon serving memakai validasi, bukan test/OOF atau oracle.
    export = (ROOT/'scripts/fusion_export_cell.py').read_text(encoding='utf-8')
    cells.insert(39, nbformat.v4.new_markdown_cell('## Pemilihan validasi dan ekspor bundle Streamlit\nOracle dilarang menjadi calon deployment. Mock selalu ditandai tidak layak deployment.'))
    cells.insert(40, nbformat.v4.new_code_cell(export))
    result = []
    heading = 'Persiapan'
    for i, cell in enumerate(cells):
        cell.outputs = [] if cell.cell_type == 'code' else cell.get('outputs', [])
        if cell.cell_type == 'code':
            cell.execution_count = None
            if i != 1:
                source = cell.source
                tree = ast.parse(source)
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = tree.body[-1]
                    segment = ast.get_source_segment(source, last)
                    lines = source.splitlines()
                    lines[last.lineno-1:last.end_lineno] = ['display(' + segment + ')']
                    source = '\n'.join(lines)
                cell.source = "with recorder.stage(" + repr(f'Tahap {i:02d} - {heading}') + "):\n" + textwrap.indent(source, '    ')
        else:
            cell.pop('outputs',None)
            for line in cell.source.splitlines():
                if line.startswith('#'):
                    heading=line.lstrip('# ').strip()
                    break
        result.append(cell)
    nb.cells = result
    nb.metadata.kernelspec = {'name':'python3','display_name':'Python 3','language':'python'}
    nb.metadata['colab'] = {'name':'Malik_AdaptiveFusion_BGE_E5.ipynb','provenance':[]}
    nb.metadata['accelerator'] = 'GPU'
    nbformat.write(nb, ROOT/'notebooks/Malik_AdaptiveFusion_BGE_E5_Colab.ipynb')


if __name__ == '__main__':
    build()
