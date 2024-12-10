
### TODO
ML
* perplexity? / zip compression for randomness? run_sim but dont print choices, print perplexity etc instead
* grad and train
* force prefix

ML / physical? problems:
* low perplexity, BOO state is 50% L/R, and looks like ball will almost never fall far from init pos

Bugs
* some weird step bug: python pboard.py  --force-np-random -r -n2 -i 0.0 
* -vvv breaks render

Refactor
* logical sim, disentangle from debug/render
* -rr render should show "sampling mask" and "physically achievale states"

Physical board
* boundaries
* multilayer board
    * switch layers from off boundaries
    * or just "drop down on next layer" essentially "holes". How to make them trainable? 
    * some "jump forward" holes too?


