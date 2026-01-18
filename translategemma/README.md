# TranslateGemma 支持语言与使用

支持语言（WMT24++ 55 种）

语言代码参考（ISO 639-1）：
```
https://en.wikipedia.org/wiki/ISO_639-1
```

TranslateGemma 模型页（Ollama）：
```
https://ollama.com/library/translategemma
```

支持语言列表（更易读版）

| 代码 | 语言 | 地区 | 类别 |
| --- | --- | --- | --- |
| cs_CZ | 捷克语 | 捷克 | WMT24 原有 |
| de_DE | 德语 | 德国 | WMT24 原有 |
| es_MX | 西班牙语 | 墨西哥 | WMT24 原有 |
| hi_IN | 印地语 | 印度 | WMT24 原有 |
| ja_JP | 日语 | 日本 | WMT24 原有 |
| ru_RU | 俄语 | 俄罗斯 | WMT24 原有 |
| uk_UA | 乌克兰语 | 乌克兰 | WMT24 原有 |
| zh_CN | 中文（简体） | 中国 | WMT24 原有 |
| ar_EG | 阿拉伯语 | 埃及 | 新增 |
| ar_SA | 阿拉伯语 | 沙特 | 新增 |
| bg_BG | 保加利亚语 | 保加利亚 | 新增 |
| bn_IN | 孟加拉语 | 印度 | 新增 |
| ca_ES | 加泰罗尼亚语 | 西班牙 | 新增 |
| da_DK | 丹麦语 | 丹麦 | 新增 |
| el_GR | 希腊语 | 希腊 | 新增 |
| et_EE | 爱沙尼亚语 | 爱沙尼亚 | 新增 |
| fa_IR | 波斯语 | 伊朗 | 新增 |
| fi_FI | 芬兰语 | 芬兰 | 新增 |
| fil_PH | 菲律宾语 | 菲律宾 | 新增 |
| fr_CA | 法语 | 加拿大 | 新增 |
| fr_FR | 法语 | 法国 | 新增 |
| gu_IN | 古吉拉特语 | 印度 | 新增 |
| he_IL | 希伯来语 | 以色列 | 新增 |
| hr_HR | 克罗地亚语 | 克罗地亚 | 新增 |
| hu_HU | 匈牙利语 | 匈牙利 | 新增 |
| id_ID | 印度尼西亚语 | 印度尼西亚 | 新增 |
| it_IT | 意大利语 | 意大利 | 新增 |
| kn_IN | 卡纳达语 | 印度 | 新增 |
| ko_KR | 韩语 | 韩国 | 新增 |
| lt_LT | 立陶宛语 | 立陶宛 | 新增 |
| lv_LV | 拉脱维亚语 | 拉脱维亚 | 新增 |
| ml_IN | 马拉雅拉姆语 | 印度 | 新增 |
| mr_IN | 马拉地语 | 印度 | 新增 |
| nl_NL | 荷兰语 | 荷兰 | 新增 |
| no_NO | 挪威语 | 挪威 | 新增 |
| pa_IN | 旁遮普语 | 印度 | 新增 |
| pl_PL | 波兰语 | 波兰 | 新增 |
| pt_BR | 葡萄牙语 | 巴西 | 新增 |
| pt_PT | 葡萄牙语 | 葡萄牙 | 新增 |
| ro_RO | 罗马尼亚语 | 罗马尼亚 | 新增 |
| sk_SK | 斯洛伐克语 | 斯洛伐克 | 新增 |
| sl_SI | 斯洛文尼亚语 | 斯洛文尼亚 | 新增 |
| sr_RS | 塞尔维亚语 | 塞尔维亚 | 新增 |
| sv_SE | 瑞典语 | 瑞典 | 新增 |
| sw_KE | 斯瓦希里语 | 肯尼亚 | 新增 |
| sw_TZ | 斯瓦希里语 | 坦桑尼亚 | 新增 |
| ta_IN | 泰米尔语 | 印度 | 新增 |
| te_IN | 泰卢固语 | 印度 | 新增 |
| th_TH | 泰语 | 泰国 | 新增 |
| tr_TR | 土耳其语 | 土耳其 | 新增 |
| ur_PK | 乌尔都语 | 巴基斯坦 | 新增 |
| vi_VN | 越南语 | 越南 | 新增 |
| zh_TW | 中文（繁体） | 台湾 | 新增 |
| zu_ZA | 祖鲁语 | 南非 | 新增 |

注：论文另提及 WMT24 的冰岛语，但未给出代码。

部署与使用（简化）

1. 启动服务

   ```bash
   docker compose -f docker/docker-compose-12b.yml build translategemma
   docker compose -f docker/docker-compose-12b.yml up -d translategemma
   ```

2. 简单 curl 示例

   ```bash
   curl -X POST "http://localhost:8020/v1/chat/completions" \
     -H "Authorization: Bearer sgl-R5tG8uI7vW9xY0zAbC2dE4fG6hJ8kL9m" \
     -H "Content-Type: application/json" \
     --data '{
       "model": "translategemma",
       "messages": [
         {
           "role": "user",
           "content": [
             {
               "type": "text",
               "source_lang_code": "en",
               "target_lang_code": "it",
               "text": "Hello, world."
             }
           ]
         }
       ]
     }'
   ```

3. Markdown 翻译（推荐）

   ```bash
   export VLLM_API_KEY=sgl-R5tG8uI7vW9xY0zAbC2dE4fG6hJ8kL9m
   python translate_markdown.py -i README.md -o README.zh.md --source-lang en --target-lang zh
   ```

   说明：
   - 必须指定 `--source-lang`
   - 代码块/行内代码/HTML/LaTeX/图片标签会被保护
   - 标题/列表/表格结构保持不变
   - `--max-chars` 控制分块长度（0=自动计算，默认自动）
   - `--max-tokens` 控制单次输出上限（默认走配置）
   - `--pool-max-workers` 控制客户端并发请求数（仅分块模式生效；整文件模式会忽略）

   示例（自动风格判定）：

   ```bash
   python translate_markdown.py -i data/input.md -o data/output_translated.md \
     --source-lang eng --target-lang zh-Hans --style-mode auto
   ```

**translate_markdown.py 实现要点**

- **分段与合并**：按段落/标题/列表合并文本，支持 `--max-chars` 自动推导，避免超出模型上下文。
- **表格策略（Cell-only）**：HTML/Markdown 表格只做单元格翻译，避免结构破坏；即使传 `--table-batch-mode` 也会强制为 `cell`。
- **格式保护**：Markdown/HTML/LaTeX/代码块/链接/图片等内容先保护占位，翻译后还原，确保结构不变。
- **提示词模板**：使用 `prompt_templates/translate_gemma_12b.txt` 渲染；可用 `--custom-system-prompt` 增加自定义指令。
- **风格控制**：支持 `--style-mode auto|fixed|off`；`auto` 会抽取样本判断风格并加入风格指令。
- **缓存**：支持持久化缓存（SQLite），命中时直接返回，避免重复请求。

4. 图片翻译（说明）

   图片翻译会将图片中的文字翻译为目标语言，输出仅包含文本，不会生成图片。

5. 纯 pipeline 版本（不依赖 vLLM）

   ```bash
   python translate_markdown_pipeline.py -i README.md -o README.zh.md --source-lang en --target-lang zh
   ```
