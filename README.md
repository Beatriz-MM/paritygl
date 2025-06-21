# ParityGL

<div align="center">
  <img src="https://github.com/Beatriz-MM/ParityGL/raw/main/assets/paritygl-logo.jpeg" width="200" alt="Logo de ParityGL">
</div>

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/downloads/release/python-310/)

###### 📚 _TFG: Análise de toxicidade en contas galegas de Instagram: desenvolvemento dun sistema de detección_  
###### 📚 _FYP: Toxicity analysis on Galician Instagram accounts: development of a detection system_

---
## :octopus:<img src="https://upload.wikimedia.org/wikipedia/commons/e/e7/Instagram_logo_2016.svg" alt="Instagram" width="18" style="vertical-align: -5px;"/> Sobre o proxecto / About the project

Este repositorio contén o traballo desenvolvido para o TFG de [Beatriz-MM](https://github.com/Beatriz-MM), que inclúe o código, os datasets orixinais e o corpus final utilizado para adestrar modelos de detección de toxicidade en lingua galega.

This repository contains the code, original datasets, and final corpus developed for the undergraduate thesis by [Beatriz-MM](https://github.com/Beatriz-MM), focused on detecting toxicity in Galician Instagram comments.

> [!NOTE]
> A descrición detallada do proxecto, metodoloxía e resultados está dispoñíbel na memoria do TFG (enlace próximamente).  
> A full project description and methodology will be available in the thesis report (coming soon).

---

## 📂 Estrutura do repositorio / Repository structure

- 📁 **/assets/** — Recursos gráficos do proxecto / Graphic resources
- 📁 **/corpus/** — Corpus final etiquetado para adestramento / Final labeled corpus for training
- 📁 **/raw_data/** — Datos brutos recollidos de Instagram (.zip por categoría) / Raw data collected from Instagram (zipped by category)
- 📁 **/scripts/** — Código dividido por tarefas / Code organized by task:
  - 📄 `corpus_conversions/` — Conversión entre formatos de corpus / Corpus format conversion
  - 📄 `data_collection/` — Recollida de datos / Data collection scripts
  - 📄 `development/` — Scripts experimentais e probas / Experimental and testing scripts
  - 📄 `emoji_analysis/` — Análise de emojis nos comentarios / Emoji analysis
  - 📄 `preprocessing/` — Limpeza e preparación textual / Text cleaning and preprocessing



## 📦 Instalación / Installation

Instala as dependencias executando: / Install all required dependencies with:

```bash
pip install -r requirements.txt
```

## 💬 Citá / Citation

Se este traballo che resulta útil, agradécese que o cites.
If you find this work helpful, please consider citing it.

    (BibTeX ou referencia aparecerá aquí se é necesario máis adiante)

## ⚠️ Aviso / Disclaimer

> Este repositorio contén datos recollidos de redes sociais e pode incluír contido sensible ou ofensivo.  
> This repository contains social media data and may include sensitive or offensive content.  
>  
> O uso deste material está restrinxido a fins de investigación ou educativos.  
> Use of this material is restricted to research or educational purposes only.

## 🔗 Ligazóns / Links

    📄 Publicación do TFG (próximamente)

    📧 Contacto: bmolinamuniz94@gmail.com

    📸 Instagram: @paritygl

## 🛡️ Licenza / License

Este proxecto utiliza a **Mozilla Public License 2.0 (MPL-2.0)** para todo o código fonte.

Porén, os arquivos creados pola autora do proxecto están tamén suxeitos á **Commons Clause License Condition v1.0**, que restrinxe o dereito a usar o software con fins comerciais.

Isto significa que:  
- Podes **usar**, modificar e compartir o código para fins persoais, académicos ou non comerciais.  
- Non podes **vender**, redistribuír ou ofrecer o software como parte dun produto ou servizo comercial sen permiso explícito.

➡️ Para máis detalles, consulta o arquivo [`COMMONS-CLAUSE.txt`](./COMMONS-CLAUSE.txt) ou visita [https://commonsclause.com](https://commonsclause.com).

---

This project uses the **Mozilla Public License 2.0 (MPL-2.0)** for all source code.

However, files authored by the project creator are also subject to the **Commons Clause License Condition v1.0**, which restricts the right to use the software for commercial purposes.

This means that:  
- You **can** use, modify, and share the code for personal, academic, or non-commercial purposes.  
- You **cannot** sell, redistribute, or offer the software as part of a commercial product or service without explicit permission.

➡️ For more details, see the [`COMMONS-CLAUSE.txt`](./COMMONS-CLAUSE.txt) file or visit [https://commonsclause.com](https://commonsclause.com).





