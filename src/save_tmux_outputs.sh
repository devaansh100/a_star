#!/bin/bash
mkdir out
tmux list-sessions | cut -d: -f1 | while read -r session_name; do
    # Define the output directory for the session
    session_output_file="out/$session_name.txt"
    touch "$session_output_file"
    windows=$(tmux list-windows -t "$session_name" -F "#{window_index}")
    # Iterate over each window
    for window_index in $windows; do
        # Get list of panes in the window
        panes=$(tmux list-panes -t "$session_name:$window_index" -F "#{pane_index}")
        # Iterate over each pane in the window
        for pane_index in $panes; do
            # Capture the scrollback buffer of the pane
            tmux select-pane -t "$session_name:$window_index.$pane_index"
            tmux capture-pane -t "$session_name:$window_index.$pane_index" -S - -E - | tmux save-buffer - >> "$session_output_file"
        done
    done
    echo "Session '$session_name', output exported to '$session_output_file'"
done

