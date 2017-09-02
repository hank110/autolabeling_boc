import create_boc as boc
import configuration as conf

def main():
    for ed in conf.dimensions:
        for ec in conf.num_concepts:
            boc.create_boc(conf.document,ed,conf.context,conf.min_freq,ec)

if __name__=="__main__":
    main()
