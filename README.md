# image2ppt-vision 0.2.1

独立的页面视觉分析与重建服务。输入图片，输出文字、几何候选、关系、分割掩码、素材及诊断证据。0.2.0 引入由源图哈希绑定的声明式重建计划驱动的 `reconstruct`，输出 SlideIR、局部素材、authoring 和重建报告。0.2.1 修复原生文字删除后可能留下的抗锯齿残影，以及模块低对比背景像素遮挡原生底形状改色的问题；具体范围见下文。PPTX 仍交给宿主机已有的固定 Node/Artifact 编译器；Python HTTP 服务不编译 PPTX。

分析和重建不自动调用外部大模型。可由当前 Codex 会话编写计划，或明确调用可选的 `plan` 命令生成待复核计划。

## 重建和可选模型计划

使用已有的分析结果与计划，避免重复 OCR：

```text
image2ppt-vision reconstruct --input page.png --analysis analysis.json --plan plan.json --output new-reconstruction
```

省略 `--analysis` 时，执行同一个本地任务队列/工作进程中的几何分析和 OCR，再进入重建；`--output` 必须是不存在的新目录。输出包含 `slideir.json`、`authoring.json`、`plan.json`、`source/`、`assets/` 和 `reconstruction.json`。原始图片及所有旧输出保留。

计划格式是 `ReconstructionPlan` / `1.0`，包括原始输入文件的 `source_sha256`、EXIF 校正后的 `image_size`、`author`、互不重叠的 `regions`，以及可选的 `text_overrides`、`retain_raster_ids` 和 `shapes`。区域类型为 panel/photo/diagram/header/footer；接壤允许，越界、重叠和重复 ID 拒绝。文字修订必须引用本次 OCR 的真实 ID；可附 `expected_text` 防止 OCR 证据变化后错误套用修订。`retain_raster_ids` 明确保留误识别图标或混合框的源像素。图形提案不能关闭像素验证。计划只描述图片区域和对象，不接受代码、路径或远程素材字段。加载计划不代表已验证视觉准确性。

需要外部模型参与时，明确配置 `VISION_LLM_BASE_URL`、`VISION_LLM_MODEL`、`VISION_LLM_API_KEY`，再执行：

```text
image2ppt-vision plan --input page.png --analysis analysis.json --output new-plan.json
```

此命令会将原图及精简 OCR 证据发送到配置的 OpenAI-compatible Chat Completions 接口，返回严格校验的计划。它没有默认服务商，也不在重建失败时自动触发。只允许 HTTPS，回环地址可用 HTTP；重定向、超限响应、无效 JSON、未知 OCR ID、来源哈希不匹配均拒绝。重试与超时有上限，错误消息不包含原始响应或凭据。旁侧的 `new-plan.json.provenance.json` 只记录哈希和必要元数据；模型返回不执行，原始请求/响应不保存。

HTTP 重建沿用 `POST /v1/jobs`，提交 `{"operation":"reconstruct","asset_id":"<32位ID>","plan":{...}}`。服务先核对计划的输入哈希与尺寸，再入队执行分析和重建；不会调用外部模型。客户端命令如下，下载目录是完整任务包，SlideIR 位于其 `reconstruction/` 子目录：

```text
image2ppt-vision --url http://127.0.0.1:18761 reconstruct --input page.png --plan plan.json --output new-remote-job
```

`/v1/capabilities` 报告 `reconstruction=true`、`reconstruction_requires_plan=true`、`pptx_compilation=false`；可选模型仅报告是否配置和显式 CLI 调用方式，不暴露密钥、地址或模型配置值。`SUCCEEDED` 表示任务完成，重建报告仍为 `REVIEW_REQUIRED`，不表示字体识别、视觉还原、PowerPoint/WPS 或人工验收通过。局部栅格素材只能移动和缩放，其内部对象不因此变成原生编辑对象。

## 一条命令转换整组图片

准备三个目录：原始 PNG、每页同名的分析 JSON、每页同名的计划 JSON。例如 `02.png` 对应 `02.json` 和 `02.json`。分析/计划目录中的 JSON 必须与 PNG 文件名一一对应，缺页或多出的 `index.json` 等文件会被拒绝。页面按 PNG 文件名排序。

```text
image2ppt-vision convert --input-dir "PPT图片" --analysis-dir "本次逐页分析" --plan-dir "本次逐页计划" --compiler-root "D:\Code\image2PPT" --output "新转换目录"
```

此正式入口将三类输入快照保存到新目录的 `inputs/`，逐页重建到 `pages/`，再调用指定本地项目的固定 `scripts/compose.mjs`，输出 PPTX、最终文件回渲染预览和质量报告到 `build/`。Node 可由 `VISION_NODE` 指向本地可执行文件，否则使用 PATH 中的 `node`。调用使用独立参数、`shell=False`，Windows 隐藏子进程窗口；计划不能改变要执行的程序或脚本。

`conversion.json` 保存页序、输入哈希、逐页对象统计和编译结果。失败时保留已经完成的页面及失败阶段，不覆盖旧结果；编译器诊断只保存长度、哈希、退出码和错误类型，避免把原文或凭据写进汇总日志。返回的 PPTX/预览路径必须位于本次 `build/` 下，并核对质量报告中的页数和 PPTX 哈希。这个命令组织完整转换流程，不等于已达到全原生编辑或完成视觉/目标应用验收。HTTP Python 服务仍只提供分析与 SlideIR 重建，整组 PPTX 编译在安装了固定编译器及 Artifact 运行时的宿主机执行。

## 安装与运行

Python 3.12。CPU 功能：`pip install .`。服务必须设置 `VISION_TOKEN`，模型目录由 `VISION_MODELS` 指定，数据目录由 `VISION_DATA` 或 `--data` 指定。GPU 分割在 Linux NVIDIA 环境执行；服务器采用独立容器，Windows 可只运行 HTTP 客户端。

模型准备是显式的部署步骤：`python tools/prepare_models.py --output <models>`。它记录来源、固定 SAM 提交与 SHA-256；生成的 manifest 和模型共同交付。正式推理检查 manifest，并禁止 OCR 隐式下载。服务器正式容器网络禁止对外连接。

```text
image2ppt-vision doctor
image2ppt-vision --data ./private-data analyze --input page.png
image2ppt-vision --data ./private-data segment --input page.png --options prompt.json
image2ppt-vision --data ./private-data extract --input page.png --mask mask.png
image2ppt-vision --data ./private-data compare --input source.png --reference preview.png
image2ppt-vision --data ./private-data serve
```

`prompt.json`：`{"mode":"prompt","bbox":[100,80,160,120]}`，也支持 `points` 与相同长度的前景/背景 `labels`（1/0）。坐标是 EXIF 校正后的原图像素，bbox 为 x/y/宽/高。自动分割不接收提示框；复杂掩码分割不是精细 alpha matting，不恢复遮挡背景。

提示模式可增加 `"crop_prompt":true`：先按 bbox 从原图裁剪，再交给 SAM，随后把掩码映射回完整原图。点坐标仍填写原图坐标，且必须落在裁剪框内。此模式将输出限制在裁剪区域；需要上下文或边缘外延时扩大框，或保持默认整图提示模式。模型内部仍执行其标准输入缩放。

## 通过 SSH 使用服务器

服务只绑定远端 `127.0.0.1:18761`；建立隧道：

```text
ssh -N -L 18761:127.0.0.1:18761 <server-alias>
image2ppt-vision --url http://127.0.0.1:18761 doctor
image2ppt-vision --url http://127.0.0.1:18761 --data ./private-results analyze --input page.png
```

客户端使用 `VISION_TOKEN` 或 `VISION_TOKEN_FILE`（只含令牌的文件），不要把令牌放到命令行或版本库。客户端上传文件、轮询任务并下载清单内产物，逐项校验 SHA-256。CLI 和 HTTP 共用 `Request` 契约及同一服务端执行器。

API：`GET /health`、`GET /v1/capabilities`、`POST /v1/assets`、`POST /v1/jobs`、`GET /v1/jobs/{id}`、`GET /v1/jobs/{id}/artifacts/{name}`。除 health 和接口文档外均要求 Bearer token。`/docs` 和 `/openapi.json` 提供接口契约。POST assets 为 multipart 字段 `file`；POST jobs 为 JSON，例如 `{"operation":"analyze","asset_id":"<32位ID>","segment":true}`。

任务状态区分 QUEUED/RUNNING/SUCCEEDED/PARTIAL/FAILED/INTERRUPTED。请求以异步任务执行，一个长期计算进程串行处理，超时只终止该子进程。模型在子进程内复用，超时后重新加载。部署只运行一个 API 实例，不使用多 Uvicorn worker。

## 产物与边界

每次任务单独保存 request/status/result/manifest 和图像产物，不覆盖旧任务。输入 PNG/JPEG/WebP，单帧，最大 40 MB、4000 万像素。剩余空间小于 20 GiB 拒绝新任务；默认不自动清理。

结果格式为 `VisionAnalysis 1.0`，与 SlideIR 分离。OCR 分数、SAM 预测 IoU 和几何启发式评分分别解释，不能互相比作统一概率。关系只表示几何包含、相交、对齐和重复间距，不表示业务语义。

分割输出二值掩码、透明 PNG、原区域及白/深/棋盘格对照；提示模式保留多个候选及分数。自动分割只给候选，不保证与 PPT 对象一一对应。无 OCR 结果时文字相交诊断为 NOT_RUN。像素差不决定是否降级为图片，也不代表已通过视觉验收。

## 基准与验收

`tools/prepare_reference.py --pptx <private.pptx> --images <rendered-pages> --output <new-directory>` 创建冻结图片清单和 OOXML 辅助标注草稿。所有草稿 `human_verified=false`。文本框不等于字形框；隐形、遮挡、混合文本和分组必须人工检查。原始 PPT 对象绝不传入识别请求。

`image2ppt-vision --data <private-runs> benchmark --suite <suite.json> --output <new-report-directory>` 逐页执行，失败后继续。人工确认标签缺失时算法指标为 NOT_RUN，不伪造准确率。确认标注后需更新套件 labels_sha256，保留旧套件。

标注 JSON：`{"human_verified":true,"objects":[{"kind":"text","bbox":[0,0,100,30],"text":"原文"}],"masks":[{"file":"target.png","sha256":"..."}]}`。提示分割每个 case 只对应一个目标，按模型最高 predicted_iou 选择结果，不能用真值反向挑选。自动分割分别报告召回、重复及疑似粘连。30 个已核对复杂目标不足时整体提示分割门槛为 NOT_RUN。

```text
python -m pytest tests -q
```

## 服务器业务切换

部署目录为 `/data/image2ppt-vision`，仅 GPU 0，通过 UUID 指定。初始 8 CPU、16 GiB 内存、串行推理；50% PyTorch allocator 预算不是硬件显存隔离。

`tools/gpu_services.py snapshot --record <new.json>` 仅记录明确授权的 t1-01 至 t1-08 和 t2 九个容器。随后 `stop --record <same.json>` 保留容器和数据、禁用它们的自动重启并正常停止。`restore --record <same.json>` 核对容器 ID 后恢复原策略和原运行状态。其他业务不修改。发现外部管理器重启原 GPU 容器时停止新服务并调查，不强杀既有进程。

部署失败先停止新服务，再恢复九个容器。无需改驱动、重启 Docker 或修改既有数据库。业务图片、标注、模型权重、运行令牌和完整运行产物都不进入公开源码包。

## 可复现构建与交付文件

`requirements-linux.lock` 约束 Linux x86_64 / Python 3.12 的直接和传递依赖；`environment.lock` 是实际安装快照，其中 `/app` 和 `/wheelhouse` 是镜像内路径，不可直接在 Windows 执行。`wheelhouse-manifest.json` 记录离线安装文件哈希，`models-manifest.json` 记录实际模型哈希及固定 SAM 源码提交。`openapi.json` 是导出的接口文档。

将交付的离线依赖目录放到源码目录下的 `wheelhouse/`，使用 `python tools/verify_wheelhouse.py --directory wheelhouse --manifest wheelhouse-manifest.json` 校验。先准备 `python:3.12.11-slim-bookworm` 基础镜像，再执行 `docker build --network=none -t image2ppt-vision:0.2.1 .`。构建过程无外网，缺文件或版本不匹配会失败，不会在线补包。源码 ZIP 不内嵌 CUDA 依赖、模型权重和私有验收图片；这些目录必须单独交付或按清单准备。

首次部署时，服务器目录准备好后执行 `python3 tools/deploy.py --release /data/image2ppt-vision/releases/v0.2.1`。脚本先跑容器测试，再执行已授权的九容器切换、GPU 模型烟雾测试和健康检查，保存 `evidence/deploy-*/` 下的记录；失败恢复原业务。脚本有生产停机副作用，只能在已获授权的目标主机执行。已有视觉服务按下文使用 `tools/upgrade.py`。

Docker 25 的 internal 网络不发布宿主机端口，因此 Compose 使用两个独立容器：`api` 仅加入 internal 网络，`ingress` 仅转发 TCP 到 api，绑定宿主机 127.0.0.1:18761。ingress 无模型目录、数据目录、令牌或 GPU 权限，额外限制为 0.25 CPU / 128 MiB。它不改变接口鉴权。SSH 服务若禁用端口转发，需管理员允许目标端口；部署脚本不修改 SSH 策略。

恢复业务时先执行 `docker compose --project-name image2ppt-vision-v1 --file <release>/compose.yaml stop`，再执行 `python3 <release>/tools/gpu_services.py restore --record <evidence>/gpu-services.json`。保留对应切换记录，不凭名称猜测原始重启策略。

工程测试中的 OOM 和低磁盘为故障注入，未通过耗尽生产资源制造故障。真实 GPU 烟雾测试、实际页面推理、工程异常测试、标注准确率及人工接受应分别阅读。`tools/http_acceptance.py` 是参考第 13 页（1280×720）的四操作集成测试，内含针对该页的提示框，不应作为通用分割客户端。


## 0.2.1 字体、校准与已有服务升级

默认 Windows 字体使用渲染器实际能解析的“微软雅黑、等线、黑体、宋体”及 Arial。Linux 部署必须提供已授权使用的字体文件和 `VISION_FONT_MANIFEST`，格式为 `{"fonts":[{"path":"relative.ttf","family":"微软雅黑","bold":false,"index":0,"sha256":"..."}]}`。路径相对清单且不得逃出目录；所有文件进行 SHA256 和可加载性校验，错误清单不会悄悄回退。交付的源码包不包含字体文件或模型权重。此服务器配置将独立字体目录 `/data/image2ppt-vision/fonts-v0.2.0` 只读挂载到 `/fonts`。

包内字体原点校准针对 Artifact Tool 2.8.94、1672×941 到 1280×720 画布，以及记录的字体文件哈希。仅在这些条件和验证过的字号范围匹配时修正 y 原点；不同字体/画布/版本会标记未校准。92 个合成文字样本的实际 PPTX 回渲染校准不等于真实页面或 PowerPoint/WPS 的验收。

已有 0.1.0 或 0.2.0 服务使用 `tools/upgrade.py`。先把新源码保存到独立的 `v0.2.1` 目录，按 `Dockerfile.upgrade` 离线构建新镜像 `image2ppt-vision:0.2.1`；该 Dockerfile 默认基于保留的 0.1.0 依赖镜像，安装 0.2.1 wheel。保留旧目录和旧镜像标签，已有 `image2ppt-vision:0.2.0` 标签不可用新构建覆盖。以下以当前运行版本为 0.2.0 为例：

```text
python3 tools/upgrade.py --release /data/image2ppt-vision/releases/v0.2.1 --previous-release /data/image2ppt-vision/releases/v0.2.0 --image image2ppt-vision:0.2.1 --font-manifest /fonts/manifest.json
```

`--previous-release` 必须填写当前运行 api/ingress 所属的实际旧目录；若仍运行 0.1.0，则填写其已记录的目录。脚本默认核对新版本为 0.2.1，只替换既有 Compose 项目的 api/ingress 两个容器。令牌保留，预检失败不切换；切换失败回到记录的旧镜像 ID。全部非目标容器的 ID 和运行状态进行前后核对，旧版本目录、镜像、任务与业务数据保留。字体目录继续使用 `/data/image2ppt-vision/fonts-v0.2.0` 的只读挂载。升级脚本不修改 SSH 策略；若目标服务器禁用端口转发，可继续通过普通 SSH 执行服务容器内的 CLI。

0.2.1 的形状提案仅对可验证的简单矩形/圆角矩形填色作原生化，其他轮廓保持图片并记录原因。语义区域中的原生背景形状不等于已经恢复其全部内部图形；图表、关系图和复杂插图的内部对象仍可能是栅格。输入为本次视觉计划时支持纠错、拆分、分组和一键转换；它不是无需计划即可保证准确的全自动识别器。

0.2.1 在文字形状拟合与平滑背景验证通过后，清理完整的已验证字形邻域，包括核心笔画周围的低透明度抗锯齿像素，并保护相邻文字所属像素。模块背景的 12/255 残差阈值仅用于浅色 `panel`：拟合底色的 RGB 各通道均不低于 225，且变化不超过 28。对符合条件的面板，与拟合背景的最大 RGB 通道差值不超过 12/255 的低对比像素归入可改色的原生底形状，避免这些像素以不透明图片遮挡后续改色；差值超过阈值的内容保留为局部栅格。彩色或较强渐变模块，以及 `header`/`footer` 区域继续使用原有的 3/255 残差阈值，防止深色渐变背景出现分块。照片和 `diagram` 区域不执行此模块背景归一化流程。

这项清理会对阈值内的背景差异进行归一化，不保证源图与输出无视觉差异。应结合最终 PPTX 回渲染和直接删除文字、修改底形状颜色的编辑探针复核；算法测试或探针通过均不等于 PowerPoint/WPS 或人工验收通过，报告继续保留 `REVIEW_REQUIRED` 与对应未验收状态。
