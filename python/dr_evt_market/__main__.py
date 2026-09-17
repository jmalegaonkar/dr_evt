################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Run the DR_EVT market command line through ``python -m dr_evt_market``."""

from .cli import main


raise SystemExit(main())
