<!-- SPDX-FileCopyrightText: 2026 aimcrazyxx -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Build do Mixar para Linux no Kaggle

`build_mixar_linux.sh` compila a versão mais recente do Mixar oficial
([Mixar-AI/mixar-app](https://github.com/Mixar-AI/mixar-app), branch `main`) para
Linux x86_64, aplica os patches listados em [`../patches/series`](../patches/series)
e gera um pacote portátil `mixar-<versão>-linux-x64.tar.xz`.

O script faz tudo sozinho:

1. instala o toolchain no Ubuntu 24.04 (GCC 14, que o Blender 5.2 exige, CMake, Ninja, Git LFS e as bibliotecas de X11/Wayland, áudio, libcurl, OpenSSL e libsecret);
2. baixa o Mixar (`MIXAR_REF`), aplica os patches e valida que cada um ainda encaixa;
3. baixa o Blender no commit fixado pelo Mixar e as bibliotecas pré-compiladas desse commit (do
   `dpcpp`, o compilador oneAPI de 800 MB, só o runtime SYCL de ~25 MB, que o Embree exige);
4. roda o `scripts/unix/build.sh` oficial com Ninja, mostrando o progresso a cada 1%;
5. abre o app em modo headless e importa todos os módulos Python do Mixar (teste de fumaça);
6. abre o app com interface num display virtual (Xvfb + OpenGL por software), espera o Mixar
   registrar a UI, confere que a pilha de camadas de pintura registrou e salva um screenshot
   (`logs/gui_check.png`);
7. empacota em `.tar.xz` com `.sha256` e `BUILD_INFO.txt`.

## Como rodar

**No site do Kaggle.** Importe `mixar-linux-build.ipynb` (File → Import Notebook), ligue
*Settings → Internet* e clique em *Save Version → Save & Run All*. CPU basta; leva de 3 a 4 h
(o limite do Kaggle é 12 h). O resultado aparece na aba *Output* da versão.

**Pela linha de comando** (precisa de `pip install kaggle` e do token da API em
`KAGGLE_API_TOKEN` ou `~/.kaggle/access_token`):

```bash
kaggle/push_kernel.sh --wait --download saida/
```

`--ref` escolhe a branch deste repositório que o notebook clona, `--mixar-ref` a versão do
Mixar (`main`, `latest-tag` ou uma tag como `v4.3.2`) e `--slug` o notebook (padrão
`mixar-linux-build`).

O Kaggle cancela uma versão enviada enquanto outra versão do mesmo notebook ainda roda
(*Canceled by backend*, poucos minutos depois de começar). Por isso o script se recusa a
enviar nesse caso: espere a execução terminar, cancele-a no site ou use outro `--slug`
(`--force` envia mesmo assim).

**Pelo GitHub Actions.** Adicione o token do Kaggle como secret `KAGGLE_API_TOKEN` do
repositório (Settings → Secrets and variables → Actions) e rode o workflow
*Kaggle Linux build*. Ele dispara o notebook, espera e anexa o pacote como artifact.

**Em qualquer Ubuntu 24.04** (root ou sudo, ~15 GB livres): `kaggle/build_mixar_linux.sh`.
Fora do Kaggle o pacote vai para `/tmp/mixar-build/out` e o ccache fica ligado, então
recompilar depois de mudar um patch leva minutos.

## Parâmetros (variáveis de ambiente)

| Variável | Padrão | |
|---|---|---|
| `MIXAR_REF` | `main` | `main` (mais recente), `latest-tag` ou qualquer tag/branch/commit |
| `MIXAR_ENV` | `Prod` | `Prod` aponta para `api.mixar.app` |
| `MIXAR_CUDA` | `0` | `1` compila os kernels CUDA do Cycles (exige CUDA toolkit; não existe nas sessões CPU) |
| `PATCH_SERIES` | `patches/series` | lista de patches, um por linha |
| `WORK_DIR` / `OUT_DIR` | `/tmp/mixar-build` / `/kaggle/working` | |
| `BUILD_CORES` | `nproc` | |
| `SMOKE_TEST` | `1` | `0` pula o teste de fumaça |
| `GUI_CHECK` | `1` | `0` pula a verificação com interface |

## Usando o pacote

```bash
tar -xf mixar-4.3.2-linux-x64.tar.xz
./mixar-4.3.2-linux-x64/mixar
```

Requer uma distribuição de 2024 em diante (glibc 2.39 e libstdc++ do GCC 14: Ubuntu 24.04+,
Debian 13+, Fedora 40+, Arch). O Cycles renderiza na CPU (sem CUDA/OptiX/oneAPI); o viewport
usa a GPU normalmente.

## Limitações

- O Kaggle só roda Linux, e o Blender não compila para Windows a partir do Linux. O build
  de Windows é o workflow *Windows build (newest Mixar)*
  (`.github/workflows/windows-build-latest.yml`): mesmo Mixar e mesmos patches, num runner
  Windows do GitHub Actions (~2h30); o resultado sai como artifact da execução.
- O código público do Mixar não traz o pacote privado `paint/procedural_materials`, mas um
  terço do módulo de pintura o importa: sem ele a pintura em camadas nem registra. O patch
  `paint-procedural-materials-fallback.patch` põe um substituto no lugar, então as camadas
  normais funcionam; a biblioteca de materiais procedurais do backend e a geração de
  materiais por IA (MatGen) aparecem como indisponíveis, e nenhum script de material é
  executado.
- O pacote é para uso pessoal. Os logos e ícones do Mixar não são GPL
  ([TRADEMARKS.md](https://github.com/Mixar-AI/mixar-app/blob/main/TRADEMARKS.md)): não
  redistribua o build sem removê-los ou substituí-los.
