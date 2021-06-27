for PARAM1 in 20; do for RHO in 100; do for DIM in 4; do for NT in exponential 1 0.5 0.1; do
#
export PARAM1 RHO DIM NT
#
sbatch -o outputs/oSMG_${PARAM1}_${RHO}_${DIM}_${NT}.stdout.txt \
-e outputs/eSMG_${PARAM1}_${RHO}_${DIM}_${NT}.txt \
--job-name=SMG_${PARAM1}_${RHO}_${DIM}_${NT} \
smg.sh
#
sleep 1
done
done
done
done
