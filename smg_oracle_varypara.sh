for PARAM1 in 10; do for RHO in 100; do for NT in 0; do for COR in 0; do
#
export PARAM1 RHO NT COR 
#
sbatch -o outputs/oSMG_${PARAM1}_${RHO}_${NT}.stdout.txt \
-e outputs/eSMG_${PARAM1}_${RHO}_${NT}.txt \
--job-name=SMG_${PARAM1}_${RHO}_${NT} \
smg_oracle.sh
#
sleep 1
done
done
done
done
