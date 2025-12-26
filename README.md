# ParityGL

<div align="center">
  <img src="https://github.com/Beatriz-MM/ParityGL/raw/main/assets/paritygl-logo.jpeg" width="200" alt="Logo de ParityGL">
</div>

<br>

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/downloads/release/python-310/)
[![NLP](https://img.shields.io/badge/Domain-NLP-blue)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://www.mozilla.org/en-US/MPL/2.0/)
[![Commons Clause](https://img.shields.io/badge/License-Commons_Clause-red.svg)](https://commonsclause.com/)

###### 📚 _TFG: Análise de toxicidade en contas galegas de Instagram: desenvolvemento dun sistema de detección_  
###### 📚 _FYP: Toxicity analysis on Galician Instagram accounts: development of a detection system_

---
## :octopus:<img src="https://upload.wikimedia.org/wikipedia/commons/e/e7/Instagram_logo_2016.svg" alt="Instagram" width="18" style="vertical-align: -5px;"/> Sobre o proxecto / About the project

Este repositorio contén o código, os datasets orixinais e o corpus final desenvolvido para o TFG de [Beatriz-MM](https://github.com/Beatriz-MM), centrado na detección de toxicidade nos comentarios en galego de Instagram, con especial atención aos de carácter misóxino.
Este traballo toma como punto de partida o proxecto [GalMisoCorpus2023](https://github.com/luciamariaalvarezcrespo/GalMisoCorpus2023), avaliando os seus modelos no contexto de Instagram e desenvolvendo novas solucións adaptadas.


This repository contains the code, the original datasets, and the final corpus developed for the Bachelor's Thesis of [Beatriz-MM](https://github.com/Beatriz-MM), focused on the detection of toxicity in Galician Instagram comments, with special attention to those of a misogynistic nature.
This work builds upon the [GalMisoCorpus2023](https://github.com/luciamariaalvarezcrespo/GalMisoCorpus2023) project by evaluating its models in the context of Instagram and developing new solutions adapted to this platform.

> [!NOTE]
> A descrición detallada do proxecto, metodoloxía e resultados está dispoñíbel na memoria do TFG.  
> A detailed description of the project, methodology, and results is available in the thesis report.
> 
> [Ligazón/Link](https://hdl.handle.net/2183/45567)

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

<br>

## 📦 Instalación / Installation

Instala as dependencias executando: / Install the dependencies by running:
```bash
pip install -r requirements.txt
```
> [!NOTE]
> 💡 Usa unha contorna virtual en Linux para evitar conflitos. / Use a virtual environment on Linux to avoid conflicts.
> 
> O ficheirto atópase en / The file is located in   
> https://github.com/luciamariaalvarezcrespo/GalMisoCorpus2023/blob/main/requirements.txt

---

## 💬 Citá / Citation

Se este traballo che resulta útil, agradécese que o cites. / If you find this work helpful, please consider citing it. 💖

```bibtex
    @misc{molina2025analise,
      author       = {Beatriz Molina Muñiz},
      title        = {Análise de toxicidade en contas galegas de Instagram: desenvolvemento dun sistema de detección},
      year         = {2025},
      howpublished = {Trabajo Fin de Grao, Universidade da Coruña},
      url          = {https://hdl.handle.net/2183/45567}
    }
```
<br>

## ⚠️ Aviso / Disclaimer

> Este repositorio contén datos recollidos de redes sociais e pode incluír contido sensible ou ofensivo.  
> O uso deste material está restrinxido a fins de investigación ou educativos.
> 
> This repository contains social media data and may include sensitive or offensive content.    
> Use of this material is restricted to research or educational purposes only.

---

## 🔗 Ligazóns / Links

  📄 [Publicación do TFG](https://hdl.handle.net/2183/45567)

  📧 bmolinamuniz94@gmail.com

  📸 Instagram: @paritygl

---  

## 🛡️ Licenza / License

Este proxecto utiliza a **Mozilla Public License 2.0 (MPL-2.0)** para todo o código fonte.

Porén, os arquivos creados pola autora do proxecto están tamén suxeitos á **Commons Clause License Condition v1.0**, que restrinxe o dereito a usar o software con fins comerciais.

Isto significa que:  
- Podes **usar**, modificar e compartir o código para fins persoais, académicos ou non comerciais.  
- Non podes **vender**, redistribuír ou ofrecer o software como parte dun produto ou servizo comercial sen permiso explícito.

➡️ Para máis detalles, consulta o arquivo [`COMMONS-CLAUSE.txt`](./COMMONS-CLAUSE.txt) ou visita [https://commonsclause.com](https://commonsclause.com).

<br>
:gb:
This project uses the **Mozilla Public License 2.0 (MPL-2.0)** for all source code.

However, files authored by the project creator are also subject to the **Commons Clause License Condition v1.0**, which restricts the right to use the software for commercial purposes.

This means that:  
- You **can** use, modify, and share the code for personal, academic, or non-commercial purposes.  
- You **cannot** sell, redistribute, or offer the software as part of a commercial product or service without explicit permission.

➡️ For more details, see the [`COMMONS-CLAUSE.txt`](./COMMONS-CLAUSE.txt) file or visit [https://commonsclause.com](https://commonsclause.com).





