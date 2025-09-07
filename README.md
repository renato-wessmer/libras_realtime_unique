# LIBRAS Realtime — LSTM (dinâmicos) + RandomForest (estáticos)

Projeto base para **reconhecer sinais de LIBRAS em tempo real** usando:
- **MediaPipe + OpenCV** para captura de landmarks
- **RandomForest (scikit-learn)** para **gestos estáticos**
- **LSTM (PyTorch)** para **gestos dinâmicos**
- **NumPy / Pandas** para manipulação de dados
- **Matplotlib** para plots
- **statsmodels (opcional)** para suavização
- **pyttsx3** para TTS (fala em pt-BR offline)

> Este repositório é um *esqueleto funcional* para você treinar com seus próprios dados.
> Ajuste os **rótulos** em `label_map.json`, grave amostras com `collect_data.py`,
> treine com `train_static.py` e `train_dynamic.py`, e rode `infer_realtime.py` para inferência ao vivo.

## Estrutura
```
libras_realtime/
  data/
    static/<label>/*.npy        # amostras estáticas (1 frame -> 1 vetor de features)
    dynamic/<label>/<seq_id>/*.npy  # sequências (N frames -> N vetores)
  models_ckpt/
    rf_static.joblib
    lstm_dynamic.pt
  config.py
  features.py
  collect_data.py
  train_static.py
  train_dynamic.py
  infer_realtime.py
  evaluate.py
  models/
    lstm_model.py
  label_map.json
  requirements.txt
  README.md
```

## Instalação (Windows/Linux/Mac)
Crie um ambiente e instale as dependências:
```bash
python -m venv .venv
# Ative:
#   Windows: .venv\Scripts\activate
#   Linux/Mac: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Se tiver problemas com `mediapipe`, consulte a doc da sua plataforma. Em GPUs antigas, use CPU.

## Defina seus rótulos
Edite `label_map.json`. Exemplo:
```json
{
  "static": ["PAZ", "OK", "BOM", "DIA"],
  "dynamic": ["BOM_DIA", "OBRIGADO", "DESCULPA"]
}
```
- *Estáticos*: posição de mão/pose não muda significativamente por alguns frames.
- *Dinâmicos*: há **movimento** significativo ao longo do tempo (precisam de sequência).

## Coleta de dados
Execute com webcam:
```bash
python collect_data.py --mode static --label PAZ --num 200
python collect_data.py --mode dynamic --label BOM_DIA --seq-len 30 --num 100
```
- Pressione **SPACE** para registrar uma amostra (estático) ou iniciar/parar uma sequência (dinâmico).
- Pressione **Q** para sair.

## Treinamento
Estáticos (RandomForest):
```bash
python train_static.py
```
Dinâmicos (LSTM):
```bash
python train_dynamic.py --epochs 20 --seq-len 30 --batch-size 64
```

## Inferência em tempo real
```bash
python infer_realtime.py --seq-len 30 --static-thr 0.6 --dynamic-thr 0.6
```
- Mostra o rótulo no vídeo, escreve **texto** e dispara **fala** via TTS.

## Avaliação (simples)
Depois de treinar, rode:
```bash
python evaluate.py
```

## Dicas
- Capture em **boas condições de luz**.
- Garanta **variedade** (pessoas, distâncias, roupas, fundos).
- Normalize os dados (já feito no `features.py`).
- Para Blender/sintético: exporte vídeos ou keypoints e alimente `features.py` para gerar .npy.

Boa pesquisa! ✨


## Fluxo rápido: treinar **1 palavra** a partir de **3 vídeos**

Suponha que você quer treinar a palavra **MEU** (dinâmica) usando 3 vídeos.

1. Coloque os vídeos em `data/raw/`:
```
data/raw/vid1.mp4
data/raw/vid2.mp4
data/raw/vid3.mp4
```

2. Edite `label_map.json`:
```json
{
  "static": ["NONE"],
  "dynamic": ["MEU", "NONE"]
}
```

> **Por quê `NONE`?** Precisamos de uma classe negativa para o treino (frames/trechos sem a palavra).

3. Ingerir (rotular) a partir dos vídeos:
- Abra cada vídeo e **pressione `S`** para marcar quando a palavra **MEU** está sendo feita; pressione `S` novamente para sair da marcação. **`Q`** sai.
- Trechos marcados viram **positivos**; o restante vira **negativo** (com amostragem).
```bash
python ingest_video.py --video data/raw/vid1.mp4 --mode dynamic --label MEU --seq-len 30 --write-neg
python ingest_video.py --video data/raw/vid2.mp4 --mode dynamic --label MEU --seq-len 30 --write-neg
python ingest_video.py --video data/raw/vid3.mp4 --mode dynamic --label MEU --seq-len 30 --write-neg
```

4. Treinar LSTM:
```bash
python train_dynamic.py --epochs 15 --seq-len 30 --batch-size 64 --hidden 128 --layers 2
```

5. Rodar em tempo real (UI com sidebar, landmarks e frase no topo):
```bash
python infer_realtime.py --seq-len 30 --dynamic-thr 0.6 --no-tts
```

> Se sua palavra for **estática**, use `--mode static` ao ingerir:
> ```bash
> python ingest_video.py --video data/raw/vid1.mp4 --mode static --label PAZ --write-neg
> python train_static.py
> python infer_realtime.py --static-thr 0.6
> ```


### Frases na tela (ex.: escrever **Bom dia** quando reconhecer **DIA**)
Edite `phrase_map.json` para mapear rótulos reconhecidos para frases exibidas e faladas:
```json
{
  "DIA": "Bom dia"
}
```
Isso faz com que, ao detectar a classe `DIA`, a UI escreva **Bom dia** no topo e também fale **bom dia** (se o TTS estiver ligado).
