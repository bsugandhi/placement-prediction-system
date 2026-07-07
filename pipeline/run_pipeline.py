"""
Pipe and Filter Architecture Pattern Implementation
=====================================================
- Filters: Independent processing components that transform data
- Pipes: Connectors that pass data (output of one filter → input of next)

Each filter:
- Is independent and self-contained
- Reads from an input pipe, processes, writes to output pipe
- Has no knowledge of other filters in the system
- Can be replaced, reordered, or tested independently
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.data_ingestion import ingest_data
from pipeline.data_cleaning import clean_data
from pipeline.feature_engineering import engineer_features
from pipeline.model_training import train_models
from pipeline.model_evaluation import evaluate_and_select


class Pipe:
    """Connector between filters. Holds data flowing between processing stages."""

    def __init__(self, data=None):
        self.data = data

    def write(self, data):
        self.data = data

    def read(self):
        return self.data


class Filter:
    """Independent processing component in the Pipe and Filter architecture."""

    def __init__(self, name: str, process_fn):
        self.name = name
        self.process_fn = process_fn

    def execute(self, input_pipe: Pipe, output_pipe: Pipe):
        """Read from input pipe, process, write to output pipe."""
        input_data = input_pipe.read()
        print(f"[Filter: {self.name}] Processing...")
        result = self.process_fn(input_data)
        output_pipe.write(result)
        print(f"[Filter: {self.name}] Done.")
        return output_pipe


class PipeAndFilterSystem:
    """Pipe and Filter architecture: data flows through independent filters
    connected by pipes.

    Pattern characteristics:
    - Filters are decoupled - they don't know about each other
    - Pipes connect filters and transfer data
    - Filters can be added, removed, or reordered independently
    - Each filter has a single responsibility
    """

    def __init__(self):
        self.filters = []

    def add_filter(self, name: str, process_fn):
        """Add a filter to the system."""
        self.filters.append(Filter(name, process_fn))

    def run(self, initial_input):
        """Execute all filters in sequence, connected by pipes."""
        print("\n" + "=" * 60)
        print("PIPE AND FILTER SYSTEM - Placement Prediction")
        print("=" * 60)

        # Create initial pipe with input data
        current_pipe = Pipe(initial_input)

        for i, filt in enumerate(self.filters, 1):
            print(f"\n{'─' * 60}")
            print(f"Filter {i}/{len(self.filters)}: {filt.name}")
            print(f"{'─' * 60}")

            # Create output pipe for this filter
            output_pipe = Pipe()

            # Execute filter: reads from current_pipe, writes to output_pipe
            filt.execute(current_pipe, output_pipe)

            # Output pipe becomes input pipe for next filter
            current_pipe = output_pipe

        print(f"\n{'=' * 60}")
        print("ALL FILTERS EXECUTED SUCCESSFULLY")
        print(f"{'=' * 60}\n")

        return current_pipe.read()


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "placement_data.csv"
    )

    # Build the Pipe and Filter system
    system = PipeAndFilterSystem()
    system.add_filter("Data Ingestion", ingest_data)
    system.add_filter("Data Cleaning", clean_data)
    system.add_filter("Feature Engineering", engineer_features)
    system.add_filter("Model Training", train_models)
    system.add_filter("Model Evaluation", evaluate_and_select)

    # Run: data flows through pipes connecting each filter
    result = system.run(data_path)
