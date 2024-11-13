# BEHAVIOR='hallucination'
LAYERS='12 13 14'
MULTIPLIERS='-1.5 -1 0 1 1.5'
# python generate_vectors.py --layers $LAYERS --save_activations --model_size "7b" --behaviors $BEHAVIOR
# python normalize_vectors.py --layers $LAYERS --model_size "7b" --behaviors $BEHAVIOR
# python prompting_with_steering.py --behaviors $BEHAVIOR --layers $LAYERS --multipliers $MULTIPLIERS --type open_ended --model_size "7b" --overwrite
# python scoring.py --behaviors $BEHAVIOR --overwrite --do_printing
# python plot_results.py --behaviors $BEHAVIOR --layers $LAYERS --multipliers $MULTIPLIERS --type open_ended

BEHAVIOR='refusal'
# python generate_vectors.py --layers $LAYERS --save_activations --model_size "7b" --behaviors $BEHAVIOR
# python normalize_vectors.py --layers $LAYERS --model_size "7b" --behaviors $BEHAVIOR
# python prompting_with_steering.py --behaviors $BEHAVIOR --layers $LAYERS --multipliers $MULTIPLIERS --type open_ended --model_size "7b" --overwrite
# python scoring.py --behaviors $BEHAVIOR --overwrite --do_printing
python plot_results.py --behaviors $BEHAVIOR --layers $LAYERS --multipliers $MULTIPLIERS --type open_ended