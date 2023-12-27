import sys
import argparse
import random
import io
import os
import tqdm

"""
take queries (x), gt (y), predictions (pred), costs (cost) and subword_len(if not like QB, then put 1) from stdin
then filter output based on threshold

"""


def main():
    thresholds = [0.005 * i for i in range(1, 200, 2)]
    missed = [0] * len(thresholds)
    exact = [0] * len(thresholds)
    prefix_exact = [0] * len(thresholds)
    matched_length = [0] * len(thresholds)
    prefix_matched_length = [0] * len(thresholds)
    tot_lines = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(args.output):
        with open(args.output, "w") as f:
            f.write(
                "thres;trigger_rate;synctatic_match;Psynctatic_match;matched_length;Pmatched_length;saved_keystrokes;Psaved_keystrokes\n"
            )
    with io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8") as f:
        for query in tqdm.tqdm(f):
            tot_lines += 1
            query = query.strip()
            if query != "\t".join(["-"] * 5):
                x, y, pred, cost, subword_len = query.split("\t")
                x = x.lower()
                y = y.lower()
                pred = pred.lower()
                y = y.strip()
                pred = pred.strip()
                #print(f"{x}\t{y}\t{pred}\t{cost}")
                cost = float(cost)
                subword_len = int(subword_len)
            for i, threshold in enumerate(thresholds):
                if query == "\t".join(["-"] * 5):
                    missed[i] += 1
                    continue
                if cost > threshold:
                    if pred == y:
                        #print("hello", len(pred))
                        exact[i] += 1
                        matched_length[i] += len(y)
                    if y.startswith(pred) and (len(pred) > 0):
                        #print("bye", len(pred))
                        prefix_exact[i] += 1
                        prefix_matched_length[i] += len(pred)
                        if len(pred) == 0:
                            print(query)
                else:
                    missed[i] += 1
    for i, threshold in enumerate(thresholds):
        with open(args.output, "a") as f:
            if exact[i] == 0:
                f.write(
                    "{};{};{};{};{};{};{};{}\n".format(threshold, 0, 0, 0, 0, 0, 0, 0)
                )
            else:
                f.write(
                    "{:.3f};{:.5f};{:.5f};{:.5f};{:.5f};{:.5f};{};{}\n".format(
                        threshold,
                        1 - missed[i] / tot_lines,
                        exact[i] / (tot_lines - missed[i]),
                        prefix_exact[i] / (tot_lines - missed[i]),
                        matched_length[i] / exact[i],
                        prefix_matched_length[i] / prefix_exact[i],
                        matched_length[i],
                        prefix_matched_length[i],
                    )
                )


if __name__ == "__main__":
    main()
