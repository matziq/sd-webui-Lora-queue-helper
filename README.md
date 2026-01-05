# Lora Queue Helper for SD WebUI A1111

A script that helps you generate a batch of Lora, using the same prompt & settings.

Helpful to compare Lora for the same character with identical prompt & settings / Generate different characters from the same source (like the sample picture) / Or just switch and pick Lora easily without changing the tab.


![UI screenshot](https://raw.githubusercontent.com/Yinzo/sd-webui-Lora-queue-helper/main/docs/ui.png)
![Output sample](https://raw.githubusercontent.com/Yinzo/sd-webui-Lora-queue-helper/main/docs/output_sample.png)

## Install

### This fork (Forge-focused)

To install from webui, go to Extensions -> Install from URL, paste <https://github.com/matziq/sd-webui-Lora-queue-helper> into the URL field, and press Install.

This fork adds a few quality-of-life improvements for browsing and queueing LoRAs:

- Refresh button for the LoRA list
- Filter textbox (with clear and “show checked” helpers)
- Sorting: alphabetical (toggle A–Z / Z–A), by file date, and random

### Upstream

Original project: <https://github.com/Yinzo/sd-webui-Lora-queue-helper>

If you prefer the original behavior (without the fork-specific changes), install the upstream extension instead.

## How to use

1. Locate the **Script** menu in the bottom left corner.
2. Select **"Apply on every Lora"**.
3. Select the **folder** that contains the Lora you want to use.
4. Select the **lora** you want to use.
5. Generate.

## Tips

- Place your Lora into **sub-folders** by class.
- Use built-in Lora configuration to store **Activation Text & Preferred Weight**, which will be automatically used in this script. Otherwise, it will only apply the Lora with default 1 weight.
