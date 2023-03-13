import psycopg2 as pg
import numpy as np
import csv
import os

slo_bins = [10, 15, 20, 30, 50, 80, 120, 180, 270, 410, 620, 930, 1400, 2100, 3150, 4730]
timeframe = 2
dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),str(timeframe))


if __name__ == '__main__':

    conn = pg.connect(dbname='Traces', user='adi')
    cur = conn.cursor()

    for slo_bin in slo_bins:
        cur.execute("SELECT timeframe_bin,req_count FROM bins WHERE slo_bin={} ORDER BY timeframe_bin;".format(slo_bin))
        arrivals = np.array([[x[0],x[1]] for x in cur.fetchall()], dtype=np.float32)
        cur.execute("SELECT MAX(req_count) FROM bins WHERE slo_bin={};".format(slo_bin))
        max_arrival = cur.fetchone()[0]
        arrivals[:,1] = np.round(arrivals[:,1]/max_arrival, 4)

        np.save(dir+'/load_{}.npy'.format(slo_bin), arrivals)
        with open(dir+'/load_{}.csv'.format(slo_bin),'w+') as f:
            writer = csv.writer(f)
            writer.writerow(['time_upper','rate'])
            writer.writerows(arrivals)


    cur.close()
    conn.close()
