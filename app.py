from pathlib import Path
import io, shutil, subprocess, tempfile, base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
ROOT=Path(__file__).resolve().parent; ART=ROOT/'artifacts'
app=FastAPI(title='Fingerprint Integrity Lab')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
@app.get('/',response_class=HTMLResponse)
def home(): return (ROOT / "static" / "index.html").read_text(
        encoding="utf-8"
    )
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)
def gray(raw):
 try:
  im=Image.open(io.BytesIO(raw)).convert('L'); a=np.asarray(im,dtype=np.uint8)
  if min(a.shape)<24: raise ValueError('Image is too small.')
  return a
 except Exception as e: raise HTTPException(400,f'Could not read image: {e}')
def energy(a):
 gy,gx=np.gradient(a.astype(np.float32)); return np.abs(gx)+np.abs(gy)
def seam_path(e):
 h,w=e.shape; cost=e.copy(); parent=np.zeros((h,w),np.int8)
 for y in range(1,h):
  prev=cost[y-1]; choices=np.vstack((np.r_[np.inf,prev[:-1]],prev,np.r_[prev[1:],np.inf])); arg=np.argmin(choices,axis=0)-1; parent[y]=arg; cost[y]+=choices[arg+1,np.arange(w)]
 s=np.zeros(h,np.int32); s[-1]=np.argmin(cost[-1])
 for y in range(h-1,0,-1): s[y-1]=np.clip(s[y]+parent[y,s[y]],0,w-1)
 return s
def carve_vertical(a,n):
 out=a.copy()
 for _ in range(n):
  h,w=out.shape; s=seam_path(energy(out)); mask=np.ones((h,w),bool); mask[np.arange(h),s]=False; out=out[mask].reshape(h,w-1)
 return out
def dataurl(a):
 b=io.BytesIO(); Image.fromarray(a).save(b,format='PNG'); return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
@app.post('/api/carve')
async def carve(image:UploadFile=File(...),ratio:int=Form(...)):
 if ratio not in (5,10,20): raise HTTPException(400,'Choose 5%, 10%, or 20%.')
 raw=await image.read()
 if len(raw)>15*1024*1024: raise HTTPException(413,'Maximum upload size is 15 MB.')
 a=gray(raw); h,w=a.shape; n=max(1,round(w*ratio/100))
 if n>=w-8: raise HTTPException(400,'Image is too narrow for this ratio.')
 reduced=carve_vertical(a,n); restored=np.asarray(Image.fromarray(reduced).resize((w,h),Image.Resampling.BILINEAR))
 return {'image':dataurl(restored),'ratio':ratio,'width':w,'height':h,'removed_seams':n,'note':'Backward-energy vertical seam removal followed by bilinear restoration.'}
def feature_vector(a):
 from skimage.feature import local_binary_pattern
 from scipy.ndimage import sobel
 lbp=local_binary_pattern(a,P=8,R=1,method='default'); h1=np.histogram(lbp,bins=256,range=(0,256))[0].astype(float); h1/=max(h1.sum(),1)
 gx=sobel(a.astype(np.float32),axis=1,mode='reflect'); gy=sobel(a.astype(np.float32),axis=0,mode='reflect'); mag=np.hypot(gx,gy)
 h2=np.histogram(mag,bins=64,range=(0,512))[0].astype(float); h2/=max(h2.sum(),1)
 v=np.r_[h1,h2]; return v/max(np.linalg.norm(v),1e-12)
@app.get('/api/status')
def status():
 models={str(r):(ART/f'svm_{r}.joblib').exists() for r in (5,10,20)}
 return {'models':models,'model_ready':all(models.values()),'nbis_ready':bool(shutil.which('mindtct') and shutil.which('bozorth3'))}
@app.post('/api/detect')
async def detect(image:UploadFile=File(...),ratio:int=Form(...)):
 if ratio not in (5,10,20): raise HTTPException(400,'Choose 5%, 10%, or 20%.')
 p=ART/f'svm_{ratio}.joblib'
 if not p.exists(): return {'configured':False,'ratio':ratio,'message':f'No trained model at artifacts/svm_{ratio}.joblib. Benchmark figures shown on this page are paper-reported, not predictions for this upload.'}
 try:
  import joblib
  obj=joblib.load(p); x=feature_vector(gray(await image.read())).reshape(1,-1)
  if isinstance(obj,dict):
   model=obj.get('model'); scaler=obj.get('scaler')
   if model is None: raise ValueError("Artifact dict needs a 'model' entry.")
   if scaler is not None: x=scaler.transform(x)
  else: model=obj
  pred=int(model.predict(x)[0]); 
  # score=float(np.asarray(model.decision_function(x)).ravel()[0]) if hasattr(model,'decision_function') else None
  probabilities = model.predict_proba(x)[0]

# Locate the probability corresponding to class 1 (seam-carved)
  carved_index = list(model.classes_).index(1)

  prob = float(probabilities[carved_index]) * 100
  return {'configured':True,'ratio':ratio,'label':'SEAM-CARVED' if pred==1 else 'PRISTINE','model':p.name,'Probability': f'{prob:.2f}','message':'Live model prediction.'}
 except Exception as e: raise HTTPException(500,f'Classifier error: {e}') 
import nbis
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from nbis import NbisExtractor, NbisExtractorSettings, new_nbis_extractor

from fastapi import UploadFile, File, HTTPException

@app.post("/api/afis")
async def afis(
    original: UploadFile = File(...),
    carved: UploadFile = File(...)
):
    try:
        # Read uploaded images as bytes
        original_bytes = await original.read()
        carved_bytes = await carved.read()

        if not original_bytes or not carved_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty fingerprint image."
            )
        settings = NbisExtractorSettings(
            min_quality=0.0,
            get_center=False,
            check_fingerprint=False,
            compute_nfiq2=False,
            ppi=None,
        )
        extractor = new_nbis_extractor(settings)
        # Extract minutiae using your original NBIS API
        m_original = extractor.extract_minutiae(original_bytes)
        m_carved = extractor.extract_minutiae(carved_bytes)

        # Bozorth3 matching
        score = float(m_original.compare(m_carved))
        print(score)
        return {
            "available": True,
            "score": score,
            "original_filename": original.filename,
            "carved_filename": carved.filename,
            "message": "NBIS matching completed successfully.",
            'interpretation':'below reference threshold 40' if score<40 else 'at or above reference threshold 40',
            'caveat':'Single uploaded-image comparison; not an operational AFIS false-rejection estimate.'
        }

    except HTTPException:
        raise

    except Exception as e:
        print("NBIS AFIS matching failed")
        raise HTTPException(
            status_code=500,
            detail=f"NBIS matching failed: {str(e)}"
        )
