# attention package: grouped-query attention for the Mojo/MAX
# engine (single-position CPU twin + GPU kernel, prefill loop).
# Lane B2, 2026-11-07. See attention.mojo for the seam contract
# (flat visible-order K/V stores, never a cache object) and the
# RoPE read-time/write-time note.
