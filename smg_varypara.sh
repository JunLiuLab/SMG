for PARAM1 in 20; do for RHO in 100; do for NT in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do for COR in 0.2 0.4 0.6 0.8 1.0; do
#
export PARAM1 RHO NT COR 
#
sbatch -o outputs/oSMG_${PARAM1}_${RHO}_${NT}.stdout.txt \
-e outputs/eSMG_${PARAM1}_${RHO}_${NT}.txt \
--job-name=SMG_${PARAM1}_${RHO}_${NT} \
smg.sh
#
sleep 1
done
done
done
done
