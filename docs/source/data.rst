Data
====

Trained Models
--------------
- `Bepler & Berger language model <http://cb.csail.mit.edu/cb/dscript/data/models/lm_v1.sav>`_ (`paper <https://doi.org/10.48550/arXiv.1902.08661>`_)
- `Human TT3D model <http://cb.csail.mit.edu/cb/dscript/data/models/tt3d_v1.sav>`_ (`HuggingFace (DOI) <https://doi.org/10.57967/hf/6440>`__)
- `Human Topsy-Turvy model <http://cb.csail.mit.edu/cb/dscript/data/models/topsy_turvy_v1.sav>`_ (`HuggingFace (DOI) <https://doi.org/10.57967/hf/6439>`__, **recommended**)
- `Human D-SCRIPT model  <http://cb.csail.mit.edu/cb/dscript/data/models/human_v1.sav>`_ (`HuggingFace (DOI) <https://doi.org/10.57967/hf/6441>`__, from original D-SCRIPT paper)

Pre-trained models can be loaded in the prediction script from HuggingFace hub by specifying the model name. For example, to load the Topsy-Turvy model, use `samsl/topsy_turvy_v1`.

Sample Data
-----------

Sequences
~~~~~~~~~
- `Human`_
- `Mouse`_
- `Fly`_
- `Yeast`_
- `Worm`_
- `E.coli`_

Interactions
~~~~~~~~~~~~
- `Human Train`_
- `Human Test`_
- `Mouse Test`_
- `Fly Test`_
- `Yeast Test`_
- `Worm Test`_
- `E. coli Test`_

.. _`Human`: https://github.com/samsledje/D-SCRIPT/blob/main/data/seqs/human.fasta
.. _`Mouse`: https://github.com/samsledje/D-SCRIPT/blob/main/data/seqs/mouse.fasta
.. _`Fly`: https://github.com/samsledje/D-SCRIPT/blob/main/data/seqs/fly.fasta
.. _`Yeast`: https://github.com/samsledje/D-SCRIPT/blob/main/data/seqs/yeast.fasta
.. _`Worm`: https://github.com/samsledje/D-SCRIPT/blob/main/data/seqs/worm.fasta
.. _`E.coli`: https://github.com/samsledje/D-SCRIPT/blob/main/data/seqs/ecoli.fasta
.. _`Human Train`: https://github.com/samsledje/D-SCRIPT/blob/main/data/pairs/human_train.tsv
.. _`Human Test`: https://github.com/samsledje/D-SCRIPT/blob/main/data/pairs/human_test.tsv
.. _`Mouse Test`: https://github.com/samsledje/D-SCRIPT/blob/main/data/pairs/mouse_test.tsv
.. _`Fly Test`: https://github.com/samsledje/D-SCRIPT/blob/main/data/pairs/fly_test.tsv
.. _`Yeast Test`: https://github.com/samsledje/D-SCRIPT/blob/main/data/pairs/yeast_test.tsv
.. _`Worm Test`: https://github.com/samsledje/D-SCRIPT/blob/main/data/pairs/worm_test.tsv
.. _`E. coli Test`: https://github.com/samsledje/D-SCRIPT/blob/main/data/pairs/ecoli_test.tsv
