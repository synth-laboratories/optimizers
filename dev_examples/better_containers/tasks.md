# Better Containers Tasks

## Separate Push: Taskset Vocabulary

Move the generic container data vocabulary away from dataset-specific naming:

- `dataset` -> `taskset`
- `/dataset/rows` -> `/taskset/tasks`
- `seed` and `seed_id` -> `task` and `task_id`
- `seeds` and `seed_ids` -> `task_ids`

This should be a separate push from the initial container authoring facade work.
Keep the migration focused on the public container contract, examples, and
optimizer compatibility checks that consume the generic data-access surface.
