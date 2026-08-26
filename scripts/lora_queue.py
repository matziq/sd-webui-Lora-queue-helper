import os
import json
import copy
import random
import math
import time

import gradio as gr

from modules import sd_samplers, errors, scripts, images, sd_models
from modules.processing import Processed, process_images
from modules.shared import state, cmd_opts, opts
from pathlib import Path

lora_dir = Path(cmd_opts.lora_dir).resolve()

# preserve original os.listdir and replace with a version that can sort entries by modification time or alphabetically
_orig_listdir = os.listdir
_alpha_mode = 0  # 0: A-Z, 1: Z-A
_date_mode = 1   # 0: newest first, 1: oldest first
_current_sort_type = "alpha"  # "alpha", "date", or "random"

def _listdir_sorted(path):
    try:
        entries = _orig_listdir(path)
    except Exception:
        # fall back to original behavior on error
        return _orig_listdir(path)

    try:
        if _current_sort_type == "alpha":
            if _alpha_mode == 0:  # A-Z
                entries.sort()
            else:  # Z-A
                entries.sort(reverse=True)
        elif _current_sort_type == "date":
            if _date_mode == 0:  # newest first
                entries.sort(key=lambda e: os.path.getmtime(os.path.join(path, e)) if os.path.exists(os.path.join(path, e)) else -1, reverse=True)
            else:  # oldest first
                entries.sort(key=lambda e: os.path.getmtime(os.path.join(path, e)) if os.path.exists(os.path.join(path, e)) else float('inf'), reverse=False)
        elif _current_sort_type == "random":
            random.shuffle(entries)
    except Exception:
        # if anything goes wrong, return unsorted entries
        return entries
    
    return entries

# Apply the monkey-patch so subsequent os.listdir calls return sorted entries
os.listdir = _listdir_sorted


def allowed_path(path):
    return Path(path).resolve().is_relative_to(lora_dir)

def is_directory_contain_lora(path):
    try:
        if allowed_path(path):
            safetensor_files = [f for f in os.listdir(path) if f.endswith('.safetensors')]
            return len(safetensor_files) > 0
    except FileNotFoundError:
        pass
    except Exception as e:
        print(e)

    return False

def get_directories(base_path, include_root=True):
    directories = []
    try:
        if allowed_path(base_path):
            for entry in os.listdir(base_path):
                full_path = os.path.join(base_path, entry)
                if os.path.isdir(full_path):
                    if is_directory_contain_lora(full_path):
                        directories.append(entry)
                    
                    nested_directories = get_directories(full_path, include_root=False)
                    directories.extend([os.path.join(entry, d) for d in nested_directories])

    except FileNotFoundError:
        pass
    except Exception as e:
        print(e)

    return directories

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def get_lora_name(lora_path):
    if opts.lora_preferred_name == "Filename":
        lora_name = lora_path.stem
    else:
        metadata = sd_models.read_metadata_from_safetensors(lora_path)
        lora_name = metadata.get('ss_output_name')
        # Fall back to filename if ss_output_name is missing, empty, or the literal string "None"
        # (some training tools incorrectly serialize Python's None as the string "None")
        if not lora_name or lora_name == "None":
            lora_name = lora_path.stem
    return lora_name

def get_lora_prompt(lora_path, json_path):
    # Open and read the JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extract the required fields from the JSON data
    preferred_weight = data.get("preferred weight", 1)
    activation_text = data.get("activation text", "").strip()

    try:
        if float(preferred_weight) == 0:
            preferred_weight = 1
    except:
        preferred_weight = 1

    lora_name = get_lora_name(lora_path)

    # Format the prompt string
    if activation_text:
        output = f"<lora:{lora_name}:{preferred_weight}>, {activation_text}"
    else:
        output = f"<lora:{lora_name}:{preferred_weight}>"

    return output

class Script(scripts.Script):
    sorting_priority = 10  # Add sorting priority like other Forge scripts
    
    def title(self):
        return "Apply on every Lora"

    def show(self, is_img2img):
        return scripts.AlwaysVisible  # Back to AlwaysVisible like other Forge scripts

    def ui(self, is_img2img):
        def refresh_loras(current_selected, directories, filter_text=""):
            """Refresh the LoRA list by forcing a fresh directory read"""
            global _orig_listdir
            os.listdir = _orig_listdir
            try:
                base_path = lora_dir  # Always use the default lora_dir
                all_dirs = get_directories(base_path)
                preserved_directories = [d for d in (directories or []) if d in all_dirs]

                all_loras = get_lora(base_path, preserved_directories)

                if filter_text.strip():
                    filtered_loras = [lora for lora in all_loras if filter_text.lower() in lora.lower()]
                else:
                    filtered_loras = all_loras

                # Deduplicate while preserving order so new files appear only once
                filtered_loras = list(dict.fromkeys(filtered_loras))

                visible = len(filtered_loras) > 0
                preserved_selections = [lora for lora in (current_selected or []) if lora in all_loras]

                return (
                    gr.CheckboxGroup.update(choices=all_dirs, value=preserved_directories),
                    gr.CheckboxGroup.update(choices=filtered_loras, value=preserved_selections, visible=visible),
                    gr.Button.update(visible=visible),
                    gr.Button.update(visible=visible)
                )
            finally:
                os.listdir = _listdir_sorted

        def update_dirs():
            # Temporarily restore original listdir to force fresh read
            global _orig_listdir
            os.listdir = _orig_listdir
            
            dirs = get_directories(lora_dir)  # Always use the default lora_dir
            
            # Restore the sorted listdir
            os.listdir = _listdir_sorted
            
            return gr.CheckboxGroup.update(choices=dirs, value=[])

        def show_dir_textbox_dummy():
            # This function is no longer needed but kept for compatibility
            return gr.Textbox.update(visible=False), gr.CheckboxGroup.update()

        def get_lora(base_path, directories):
            all_loras = []

            # If no directories are selected, check the base path directly
            if not directories:
                if allowed_path(base_path):
                    safetensor_files = [f for f in os.listdir(base_path) if f.endswith('.safetensors')]
                    all_loras.extend([os.path.splitext(f)[0] for f in safetensor_files])

            for directory in directories:
                directory = os.path.join(base_path, directory)
                if not allowed_path(directory):
                    continue
                safetensor_files = [f for f in os.listdir(directory) if f.endswith('.safetensors')]
                all_loras.extend([os.path.splitext(f)[0] for f in safetensor_files])

            return all_loras

        def update_loras(current_selected, directories, filter_text=""):
            base_path = lora_dir  # Always use the default lora_dir
            all_loras = get_lora(base_path, directories)
            
            # Apply filter if filter_text is provided
            if filter_text.strip():
                filtered_loras = [lora for lora in all_loras if filter_text.lower() in lora.lower()]
            else:
                filtered_loras = all_loras
            
            visible = len(filtered_loras) > 0
            # Preserve all current selections, even those not currently visible due to filtering
            # Only valid LoRAs (those that exist in all_loras) should be kept in selection
            preserved_selections = [lora for lora in current_selected if lora in all_loras]
            
            return (
                gr.CheckboxGroup.update(choices=filtered_loras, value=preserved_selections, visible=visible),
                gr.Button.update(visible=visible),
                gr.Button.update(visible=visible)
            )

        def filter_loras(filter_text, current_selected, directories):
            """Filter LoRAs based on the filter text"""
            return update_loras(current_selected, directories, filter_text)

        def clear_filter(current_selected, directories):
            """Clear the filter and return all LoRAs"""
            base_path = lora_dir  # Always use the default lora_dir
            all_loras = get_lora(base_path, directories)
            preserved_selections = [lora for lora in current_selected if lora in all_loras]
            
            return (
                gr.CheckboxGroup.update(choices=all_loras, value=preserved_selections, visible=len(all_loras) > 0),
                gr.Button.update(visible=len(all_loras) > 0),
                gr.Button.update(visible=len(all_loras) > 0),
                ""  # Clear the filter text
            )

        def show_checked_only(current_selected, directories):
            """Show only the currently checked LoRAs"""
            base_path = lora_dir  # Always use the default lora_dir
            all_loras = get_lora(base_path, directories)
            
            # Filter to show only checked LoRAs
            checked_loras = [lora for lora in current_selected if lora in all_loras]
            
            return (
                gr.CheckboxGroup.update(choices=checked_loras, value=checked_loras, visible=len(checked_loras) > 0),
                gr.Button.update(visible=len(checked_loras) > 0),
                gr.Button.update(visible=len(checked_loras) > 0)
            )

        def select_all_visible_loras(current_selected, directories, filter_text=""):
            """Select all visible LoRAs (filtered or unfiltered)"""
            base_path = lora_dir  # Always use the default lora_dir
            all_loras = get_lora(base_path, directories)
            
            # Apply filter if filter_text is provided
            if filter_text.strip():
                filtered_loras = [lora for lora in all_loras if filter_text.lower() in lora.lower()]
            else:
                filtered_loras = all_loras
            
            # Preserve existing selections not in current filter and add all visible LoRAs
            preserved_selections = [lora for lora in current_selected if lora not in filtered_loras]
            new_selected = preserved_selections + filtered_loras
            
            visible = len(filtered_loras) > 0
            return gr.CheckboxGroup.update(choices=filtered_loras, value=new_selected, visible=visible)

        def clear_all_visible_loras(current_selected, directories, filter_text=""):
            """Clear all visible LoRAs (filtered or unfiltered)"""
            base_path = lora_dir  # Always use the default lora_dir
            all_loras = get_lora(base_path, directories)
            
            # Apply filter if filter_text is provided
            if filter_text.strip():
                filtered_loras = [lora for lora in all_loras if filter_text.lower() in lora.lower()]
            else:
                filtered_loras = all_loras
            
            # Keep only selections that are not in the currently visible list
            new_selected = [lora for lora in current_selected if lora not in filtered_loras]
            
            visible = len(filtered_loras) > 0
            return gr.CheckboxGroup.update(choices=filtered_loras, value=new_selected, visible=visible)

        def sort_alphabetically(directories, filter_text="", current_selected=None):
            global _alpha_mode, _current_sort_type
            _current_sort_type = "alpha"
            _alpha_mode = (_alpha_mode + 1) % 2  # Toggle between A-Z and Z-A
            
            # Refresh the LoRA list with new sorting
            base_path = lora_dir  # Always use the default lora_dir
            all_loras = get_lora(base_path, directories)
            
            # Apply filter if filter_text is provided
            if filter_text.strip():
                filtered_loras = [lora for lora in all_loras if filter_text.lower() in lora.lower()]
            else:
                filtered_loras = all_loras
            
            visible = len(filtered_loras) > 0
            
            # Preserve selections if current_selected is provided, otherwise clear selections
            preserved_selections = []
            if current_selected is not None:
                preserved_selections = [lora for lora in current_selected if lora in all_loras]
            
            alpha_text = "📝 Alpha (A-Z)" if _alpha_mode == 0 else "📝 Alpha (Z-A)"
            
            return (
                gr.CheckboxGroup.update(choices=filtered_loras, value=preserved_selections, visible=visible),
                gr.Button.update(value=alpha_text),
                gr.Button.update(variant="secondary"),
                gr.Button.update(variant="secondary")
            )

        def _sort_by_date_mode(directories, filter_text="", current_selected=None, mode=0):
            global _date_mode, _current_sort_type
            _current_sort_type = "date"
            _date_mode = mode

            # Refresh the LoRA list with new sorting
            base_path = lora_dir  # Always use the default lora_dir
            all_loras = get_lora(base_path, directories)

            # Apply filter if filter_text is provided
            if filter_text.strip():
                filtered_loras = [lora for lora in all_loras if filter_text.lower() in lora.lower()]
            else:
                filtered_loras = all_loras

            visible = len(filtered_loras) > 0

            # Preserve selections if current_selected is provided, otherwise clear selections
            preserved_selections = []
            if current_selected is not None:
                preserved_selections = [lora for lora in current_selected if lora in all_loras]

            is_newest = mode == 0

            return (
                gr.CheckboxGroup.update(choices=filtered_loras, value=preserved_selections, visible=visible),
                gr.Button.update(variant="primary" if is_newest else "secondary"),
                gr.Button.update(variant="primary" if not is_newest else "secondary")
            )

        def sort_by_date_newest(directories, filter_text="", current_selected=None):
            return _sort_by_date_mode(directories, filter_text, current_selected, mode=0)

        def sort_by_date_oldest(directories, filter_text="", current_selected=None):
            return _sort_by_date_mode(directories, filter_text, current_selected, mode=1)

        def sort_randomly(directories, filter_text="", current_selected=None):
            global _current_sort_type
            _current_sort_type = "random"
            
            # Refresh the LoRA list with new sorting
            base_path = lora_dir  # Always use the default lora_dir
            all_loras = get_lora(base_path, directories)
            
            # Apply filter if filter_text is provided
            if filter_text.strip():
                filtered_loras = [lora for lora in all_loras if filter_text.lower() in lora.lower()]
            else:
                filtered_loras = all_loras
            
            visible = len(filtered_loras) > 0
            
            # Preserve selections if current_selected is provided, otherwise clear selections
            preserved_selections = []
            if current_selected is not None:
                preserved_selections = [lora for lora in current_selected if lora in all_loras]
            
            return (
                gr.CheckboxGroup.update(choices=filtered_loras, value=preserved_selections, visible=visible),
                gr.Button.update(value="🎲 Random"),
                gr.Button.update(variant="secondary"),
                gr.Button.update(variant="secondary")
            )

        def deselect_all_lora():
            return gr.CheckboxGroup.update(value=[])

        with gr.Column():
            gr.HTML("<style>.lora-queue-toolbar{flex-wrap:nowrap!important;gap:0.25rem;}</style>")
            with gr.Row(elem_classes=["lora-queue-toolbar"]):
                refresh_button = gr.Button("🔄 Refresh", scale=1, size="extra-small", min_width=0)
                select_all_lora_button = gr.Button("All", scale=1, size="extra-small", min_width=0)
                deselect_all_lora_button = gr.Button("Clear", scale=1, size="extra-small", min_width=0)
                alpha_sort_button = gr.Button("📝 Alpha (A-Z)", scale=1, size="extra-small", min_width=0)
                date_sort_new_button = gr.Button("📅 New→Old", scale=1, size="extra-small", variant="secondary", min_width=0)
                date_sort_old_button = gr.Button("📅 Old→New", scale=1, size="extra-small", variant="secondary", min_width=0)
                random_sort_button = gr.Button("🎲 Random", scale=1, size="extra-small", variant="secondary", min_width=0)
            
            base_dir = lora_dir  # Always use the default lora_dir
            all_dirs = get_directories(base_dir)

            directory_checkboxes = gr.CheckboxGroup(label="Select Directory", choices=all_dirs, value=[], elem_id=self.elem_id("directory_checkboxes"))

            startup_loras = get_lora(base_dir, directory_checkboxes.value)
            
            # Add filter textbox with clear button and show checked button
            with gr.Row():
                lora_filter = gr.Textbox(label="Filter LoRAs", placeholder="Type to filter LoRA names...", value="", elem_id=self.elem_id("lora_filter"))
                clear_filter_button = gr.Button("Clear Filter", scale=0, size="sm")
                show_checked_button = gr.Button("Show Checked", scale=0, size="sm")
            
            lora_checkboxes = gr.CheckboxGroup(label="Lora", choices=startup_loras, value=[], visible=len(startup_loras)>0, elem_id=self.elem_id("lora_checkboxes"))

            # Add refresh functionality
            refresh_button.click(
                fn=refresh_loras,
                inputs=[lora_checkboxes, directory_checkboxes, lora_filter],
                outputs=[directory_checkboxes, lora_checkboxes, select_all_lora_button, deselect_all_lora_button]
            )
            directory_checkboxes.change(fn=update_loras, inputs=[lora_checkboxes, directory_checkboxes, lora_filter], outputs=[lora_checkboxes, select_all_lora_button, deselect_all_lora_button])
            
            # Add filter functionality
            lora_filter.change(fn=filter_loras, inputs=[lora_filter, lora_checkboxes, directory_checkboxes], outputs=[lora_checkboxes, select_all_lora_button, deselect_all_lora_button])
            clear_filter_button.click(fn=clear_filter, inputs=[lora_checkboxes, directory_checkboxes], outputs=[lora_checkboxes, select_all_lora_button, deselect_all_lora_button, lora_filter])
            show_checked_button.click(fn=show_checked_only, inputs=[lora_checkboxes, directory_checkboxes], outputs=[lora_checkboxes, select_all_lora_button, deselect_all_lora_button])
            
            # Update button functionality to work with filtering
            select_all_lora_button.click(fn=select_all_visible_loras, inputs=[lora_checkboxes, directory_checkboxes, lora_filter], outputs=lora_checkboxes)
            deselect_all_lora_button.click(fn=clear_all_visible_loras, inputs=[lora_checkboxes, directory_checkboxes, lora_filter], outputs=lora_checkboxes)
            
            # Add sorting functionality
            alpha_sort_button.click(fn=sort_alphabetically, inputs=[directory_checkboxes, lora_filter, lora_checkboxes], outputs=[lora_checkboxes, alpha_sort_button, date_sort_new_button, date_sort_old_button])
            date_sort_new_button.click(fn=sort_by_date_newest, inputs=[directory_checkboxes, lora_filter, lora_checkboxes], outputs=[lora_checkboxes, date_sort_new_button, date_sort_old_button])
            date_sort_old_button.click(fn=sort_by_date_oldest, inputs=[directory_checkboxes, lora_filter, lora_checkboxes], outputs=[lora_checkboxes, date_sort_new_button, date_sort_old_button])
            random_sort_button.click(fn=sort_randomly, inputs=[directory_checkboxes, lora_filter, lora_checkboxes], outputs=[lora_checkboxes, random_sort_button, date_sort_new_button, date_sort_old_button])

        return [directory_checkboxes, lora_checkboxes, lora_filter]

    def process(self, p, *script_args, **kwargs):
        # Skip if this is a queued processing object
        if hasattr(p, 'lora_queue_skip'):
            return
            
        if len(script_args) >= 2:
            directories, selected_loras = script_args[0], script_args[1]
            self._modify_prompt_for_loras(p, directories, selected_loras)
        return

    def before_process_batch(self, p, *script_args, **kwargs):
        # Optional: Add progress feedback for queue processing
        if hasattr(p, 'lora_queue_info') and p.lora_queue_info:
            queue_info = p.lora_queue_info
            if len(queue_info['selected_loras']) > 1:
                current_lora = queue_info['selected_loras'][0]
                print(f"LoRA Queue Helper: Generating with {current_lora}...")
        return

    def postprocess(self, p, processed, *script_args):
        """Handle queue processing for multiple LoRAs"""
        # Skip if this is a queued processing object or already processed
        if hasattr(p, 'lora_queue_skip') or hasattr(p, 'lora_queue_processed'):
            return processed
        
        if not hasattr(p, 'lora_queue_info') or not p.lora_queue_info:
            return processed
        
        # Mark as processed to prevent recursive calls
        p.lora_queue_processed = True
        
        queue_info = p.lora_queue_info
        selected_loras = queue_info['selected_loras']
        
        # If only one LoRA, no queue processing needed
        if len(selected_loras) <= 1:
            return processed
        
        # Process remaining LoRAs
        remaining_loras = selected_loras[1:]  # Skip first LoRA (already processed)
        
        print(f"LoRA Queue Helper: Starting queue processing for {len(remaining_loras)} remaining LoRAs")
        
        # Import here to avoid circular imports
        from modules.processing import process_images
        import copy
        
        for i, lora in enumerate(remaining_loras):
            print(f"LoRA Queue Helper: Processing queue item {i+2}/{len(selected_loras)}: {lora}")
            
            try:
                # Create a new processing object for this LoRA
                p_queue = copy.copy(p)
                
                # Mark this as a queued item to prevent script from running again
                p_queue.lora_queue_skip = True
                
                # Clear any existing modifications
                if hasattr(p_queue, 'lora_helper_modified'):
                    delattr(p_queue, 'lora_helper_modified')
                if hasattr(p_queue, 'lora_queue_info'):
                    delattr(p_queue, 'lora_queue_info')
                
                # Restore original settings
                p_queue.prompt = queue_info['original_prompt']
                p_queue.n_iter = queue_info['original_n_iter']
                p_queue.batch_size = queue_info['original_batch_size']
                
                # Apply this LoRA's prompt modification directly
                modified_prompt = self._get_lora_prompt_for_single_lora(
                    queue_info['directories'], 
                    lora, 
                    queue_info['original_prompt']
                )
                p_queue.prompt = modified_prompt
                
                print(f"LoRA Queue Helper: Queue processing {lora} with {queue_info['original_n_iter']} iterations")
                
                # Reset progress state for this LoRA
                state.job_count = queue_info['original_n_iter'] * queue_info['original_batch_size']
                state.job_no = 0
                state.sampling_step = 0
                
                # Process this LoRA batch
                queue_processed = process_images(p_queue)
                
                # Add results to the main processed object
                if queue_processed and queue_processed.images:
                    processed.images.extend(queue_processed.images)
                    if hasattr(processed, 'all_prompts') and hasattr(queue_processed, 'all_prompts'):
                        processed.all_prompts.extend(queue_processed.all_prompts)
                    if hasattr(processed, 'all_negative_prompts') and hasattr(queue_processed, 'all_negative_prompts'):
                        processed.all_negative_prompts.extend(queue_processed.all_negative_prompts)
                    if hasattr(processed, 'all_seeds') and hasattr(queue_processed, 'all_seeds'):
                        processed.all_seeds.extend(queue_processed.all_seeds)
                    if hasattr(processed, 'all_subseeds') and hasattr(queue_processed, 'all_subseeds'):
                        processed.all_subseeds.extend(queue_processed.all_subseeds)
                    
                    print(f"LoRA Queue Helper: Added {len(queue_processed.images)} images for {lora}")
                else:
                    print(f"LoRA Queue Helper: No images generated for {lora}")
                    
            except Exception as e:
                print(f"LoRA Queue Helper: Error processing {lora}: {e}")
                import traceback
                traceback.print_exc()
        
        # Clear queue info
        p.lora_queue_info = None
        
        total_images = len(processed.images)
        expected_images = len(selected_loras) * queue_info['original_n_iter'] * queue_info['original_batch_size']
        print(f"LoRA Queue Helper: Queue processing complete! Generated {total_images}/{expected_images} total images")
        
        return processed

    def _modify_prompt_for_loras(self, p, directories, selected_loras):
        """Common method to modify prompt with LoRA tags - UI-based queue system"""
        # Prevent multiple modifications
        if hasattr(p, 'lora_helper_modified'):
            return
        p.lora_helper_modified = True
        
        if len(selected_loras) == 0:
            return

        # Store original prompt and settings
        if not hasattr(p, 'original_prompt_lora_helper'):
            p.original_prompt_lora_helper = p.prompt
        
        # Use the first LoRA for the initial generation
        first_lora = selected_loras[0]
        modified_prompt = self._get_lora_prompt_for_single_lora(directories, first_lora, p.original_prompt_lora_helper)
        p.prompt = modified_prompt
        
        # Also update all_prompts array for Forge compatibility
        if hasattr(p, 'all_prompts') and p.all_prompts:
            p.all_prompts = [modified_prompt] * len(p.all_prompts)
        
        print(f"LoRA Queue Helper: Starting with LoRA 1/{len(selected_loras)}: {first_lora}")
        print(f"LoRA Queue Helper: Modified prompt: {modified_prompt[:100]}...")
        
        # Store queue information for postprocess
        p.lora_queue_info = {
            'directories': directories,
            'selected_loras': selected_loras,
            'current_lora_index': 0,
            'original_prompt': p.original_prompt_lora_helper,
            'original_n_iter': p.n_iter,
            'original_batch_size': p.batch_size
        }
        
        # Print user-friendly information
        if len(selected_loras) > 1:
            total_expected = len(selected_loras) * p.n_iter * p.batch_size
            print(f"LoRA Queue Helper: Processing {len(selected_loras)} LoRAs in sequence")
            print(f"LoRA Queue Helper: Expected total images: {total_expected} ({p.n_iter} iterations × {p.batch_size} batch size × {len(selected_loras)} LoRAs)")
        else:
            print(f"LoRA Queue Helper: Processing single LoRA: {first_lora}")
            print(f"LoRA Queue Helper: Modified prompt: {modified_prompt}")
        
        print(f"LoRA Queue Helper: Current p.prompt = {p.prompt[:100]}...")
        if hasattr(p, 'all_prompts'):
            print(f"LoRA Queue Helper: all_prompts length = {len(p.all_prompts) if p.all_prompts else 0}")
            print(f"LoRA Queue Helper: Starting with LoRA 1/{len(selected_loras)}: {first_lora}")
        else:
            print(f"LoRA Queue Helper: Processing single LoRA: {first_lora}")

    def _get_lora_prompt_for_single_lora(self, directories, selected_lora, original_prompt):
        """Get the prompt with a single LoRA's tags added"""
        base_path = lora_dir
        directories_to_check = directories if directories else [""]
        
        for directory in directories_to_check:
            directory_path = base_path if directory == "" else base_path.joinpath(directory)
            if not allowed_path(directory_path):
                continue
            
            try:
                safetensor_files = [f for f in os.listdir(directory_path) if f.endswith('.safetensors')]
            except Exception:
                continue

            for safetensor_file in safetensor_files:
                lora_filename = os.path.splitext(safetensor_file)[0]
                if lora_filename != selected_lora:
                    continue
                    
                lora_file_path = directory_path.joinpath(safetensor_file)
                json_file = lora_filename + '.json'
                json_file_path = directory_path.joinpath(json_file)

                lora_tags = None
                if os.path.exists(json_file_path):
                    try:
                        lora_tags = get_lora_prompt(lora_file_path, json_file_path)
                    except Exception:
                        pass
                
                if lora_tags == None or not isinstance(lora_tags, str):
                    lora_tags = f"<lora:{lora_filename}:1>"

                # Return the modified prompt with this LoRA's tags
                modified_prompt = lora_tags + ", " + original_prompt
                return modified_prompt
        
        # If LoRA not found, return original prompt
        return original_prompt
