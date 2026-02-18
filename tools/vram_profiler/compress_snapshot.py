import json
import sys
import os

def compress_snapshot(input_path, output_path, target_size_mb=100):
    """
    Naively compresses a VRAM snapshot by reading line-by-line (if formatted) 
    or by full load if memory allows (impossible for very large files).
    
    Since PyTorch snapshots are usually one line or standard JSON, 
    we really need a stream parser. 
    
    This script assumes the user has 'ijson' installed for streaming.
    If not, it suggests installing it.
    """
    try:
        import ijson
    except ImportError:
        print("Error: This tool requires 'ijson' to handle large files.")
        print("Please run: pip install ijson")
        sys.exit(1)

    print(f"Compressing {input_path} -> {output_path}...")
    

    
    trace_events = []
    segments = []
    other_keys = {}
    
    count = 0
    kept = 0
    

    
    model_start_kept = False # Try to keep early events if found
    
    print("Scanning file stream... (This may take a while for large files)")
    
    with open(input_path, 'rb') as f:
        # 1. Parse Segments (Critical for baseline)
        # We assume 'segments' key exists.
        print("Extracting segments...")
        try:
            for item in ijson.items(f, 'segments.item'):
                segments.append(item)
        except Exception as e:
            print(f"Warning reading segments: {e}")
            f.seek(0)
            
        f.seek(0)
        
        # 2. Parse Traces & Metadata
        

        f.seek(0)
        header_data = {}
        try:
            # We want to extract small top-level fields without loading the huge array.
            # ijson.items(f, 'user_metadata') works.
            for k in ['user_metadata', 'device_names', 'device_properties', 'traceName']:
                f.seek(0)
                try:
                    for val in ijson.items(f, k):
                        header_data[k] = val
                        break # items yields the object
                except:
                    pass
        except Exception as e:
            print(f"Warning reading metadata: {e}")

        f.seek(0)
        parser = ijson.items(f, 'device_traces.item.item')
        
        for event in parser:
            count += 1
            keep = False
            
            # ALWAYS keep OOM
            if event.get('action') == 'oom':
                keep = True
            # ALWAYS keep Segment Allocs (for baseline tracking)
            elif event.get('action') in ['segment_alloc', 'segment_free']:
                keep = True
            # Keep allocs (>1MB) - changed from 10MB to catch smaller cuts
            elif event.get('size', 0) > 1 * 1024 * 1024:
                keep = True
            # Sample small events (1 in 1000)
            elif count % 1000 == 0:
                keep = True
                
            if keep:
                trace_events.append(event)
                kept += 1
            
            if count % 100000 == 0:
                sys.stdout.write(f"\rProcessed {count/1000000:.1f}M events, kept {kept}...")
                sys.stdout.flush()

    print(f"\nFinished. Kept {kept} of {count} events.")
    
    # Construct new JSON
    new_data = {
        "device_traces": [trace_events],
        "segments": segments,
        "compressed": True,
        **header_data # Include preserved metadata
    }
    
    print("Writing output...")
    with open(output_path, 'w') as f:
        json.dump(new_data, f)
    
    print(f"Done! Created {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress_snapshot.py <input_file> [output_file]")
        sys.exit(1)
        
    infile = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) > 2 else "vram_snapshot_small.json"
    
    compress_snapshot(infile, outfile)
