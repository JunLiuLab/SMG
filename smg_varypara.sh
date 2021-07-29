for PARAM1 in 2 3 4 5 6 7 8 9 10; do for RHO in 100 50; do for NT in 1.0; do for COR in 1.0; do
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
