//
// Created by root on 6/20/25.
//

#ifndef MEMORYBUDGET_H
#define MEMORYBUDGET_H

#include <atomic>

// Use this to predict the result of a HUGE_TLB mmap call. We get substantial perf improvements by keeping
// large hash sets in 1GB pages, so these are quite valuable. We assume that all 1gb pages are used by this
// program only.
static std::atomic<int> available_1gb_hugepages;

struct HugePageMapping {
    char *data;
    size_t nr_pages;

    ~HugePageMapping() {

    }
};



#endif //MEMORYBUDGET_H
