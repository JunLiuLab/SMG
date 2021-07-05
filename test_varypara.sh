for PARAM1 in 20; do for RHO in 100; do for DIM in 4; do for NT in 0.0 0.4; do for COR in 0; do for TYPE in unimodal bimodal; do
#
export PARAM1 RHO DIM NT COR TYPE
#
sbatch -o outputs/oSMG_${PARAM1}_${RHO}_${DIM}_${NT}_${COR}_${TYPE}.txt \
-e outputs/eSMG_${PARAM1}_${RHO}_${DIM}_${NT}_${COR}_${TYPE}.txt \
--job-name=SMG_${PARAM1}_${RHO}_${DIM}_${NT}_${COR}_${TYPE} \
smg.sh
#
sleep 1
done
done
done
done
done
done
