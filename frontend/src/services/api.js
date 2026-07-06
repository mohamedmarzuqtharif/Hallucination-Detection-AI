import axios from 'axios'
export const api=axios.create({baseURL:import.meta.env.VITE_API_URL||'http://localhost:8000',timeout:60000})
export const detect=(data)=>api.post('/detect',data).then(r=>r.data)
export const getHistory=()=>api.get('/history').then(r=>r.data.items)
