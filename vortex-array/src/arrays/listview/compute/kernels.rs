// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ListView;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(crate) const PARENT_KERNELS: ParentKernelSet<ListView> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(ListView))]);
