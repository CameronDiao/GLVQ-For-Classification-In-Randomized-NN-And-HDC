printf('lendo problema %s ...\n', problema);

n_entradas= 8; n_clases= 3; n_fich= 1; fich{1}= 'post-operative.data'; n_patrons(1)= 90;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j = 1:n_entradas
	  t = fscanf(f,'%s',1);
	  if t ~= '?'
		if j==1 || j==2
		  val={'low','mid','high'};
		elseif j==3
		  val={'excellent','good'};
		elseif j==4
		  val={'high','mid'};
		elseif j==5 || j==6 || j==7
		  val={'mod-stable','stable','unstable'};
		else
		  x(i_fich,i,j)=str2double(t);
		end
		n=length(val);
		for k=1:n
		  if strcmp(t, val{k})
			x(i_fich,i,j)=k; break
		  end
		end
	  else
		x(i_fich,i,j) = 0;
	  end
	end	
	t = fscanf(f,'%s',1);  	% lectura da clase
	if t=='A'
	  cl(i_fich,i)=0;
	elseif t=='I'
	  cl(i_fich,i)=1;
	elseif t=='S'
	  cl(i_fich,i)=2;
	else
	  error('clase %i descoñecida', t)
	end
  end
  fclose(f);
end
