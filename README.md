## Jax / numpy simulation of a stochastic turing tumble like game

Hopefully can be made trainable then it will be first (to our knowledge) physically realizable autoregressive transformer-decoder-ish-ly-kinda-maybe model.

https://github.com/user-attachments/assets/ee51997d-8a76-4da0-84be-f8d1fef9a624

![Image](https://github.com/user-attachments/assets/ed0e3955-227c-42a1-b857-7edccf2e557d)

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


