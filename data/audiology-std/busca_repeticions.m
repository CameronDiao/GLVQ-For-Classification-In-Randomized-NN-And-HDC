problema='audiology-std';


n_entradas= 69; n_clases= 18; 
n_fich= 2; fich{1}= 'audiology.standardized.data'; n_patrons(1)= 171; fich{2}= 'audiology.standardized.test'; n_patrons(2)= 25; %n_patrons(1)= 194; 
printf('buscando repeticions en arquivo %s ...\n', fich{1}); fflush(stdout);

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
	  t = fscanf(f,'%c',1);
	  if t ~= '?'
		fseek(f,-1,SEEK_CUR); t=fscanf(f,'%s',1);
		if j==2
		  val={'mild'; 'moderate'; 'severe'; 'normal'; 'profound'};
  		elseif j==4
  		  val={'normal';'elevated';'absent'};
		elseif j==5
		  val={'normal';'absent';'elevated'};
		elseif j==6
		  val={'mild'; 'moderate'; 'normal'; 'unmeasured'};
		elseif j==8
		  val={'normal'; 'degraded'};
		elseif j==59
		  val={'normal';'elevated';'absent'};
		elseif j==60
		  val={'normal';'absent';'elevated'};
		elseif j==64
		  val={'normal';'good';'very_good';'very_poor';'poor';'unmeasured'};
		elseif j==66
		  val={'a','as','b','ad','c'};
		else
		  val={'f', 't'}; 
		end
		n=length(val); a=2/(n-1); b=(1+n)/(1-n);
		for k=1:n
		  if strcmp(t,val{k})
			x(i_fich,i,j)=a*k+b; break
		  end
		end
%  		printf('%s ', t)
	  else  % valor ausente "?"
		fscanf(f,'%c',1); %le e descarta a coma
		x(i_fich,i,j) = 0;
%  		printf('? ')
	  end
	end
	fscanf(f,'%s',1);    % le e descarta o id de patrón
	t = fscanf(f,'%s',1);  	% lectura da clase
	if strcmp(t, 'cochlear_age') t= 0;
	elseif strcmp(t, 'cochlear_age_and_noise') t= 1;
	elseif strcmp(t, 'cochlear_noise_and_heredity') t= 2;
	elseif strcmp(t, 'cochlear_poss_noise') t= 3;
	elseif strcmp(t, 'cochlear_unknown') t= 4;
	elseif strcmp(t, 'conductive_discontinuity') t= 5;
	elseif strcmp(t, 'conductive_fixation') t= 6;
	elseif strcmp(t, 'mixed_cochlear_age_otitis_media') t= 7;
	elseif strcmp(t, 'mixed_cochlear_age_s_om') t= 8;
	elseif strcmp(t, 'mixed_cochlear_unk_discontinuity') t= 9;
	elseif strcmp(t, 'mixed_cochlear_unk_fixation') t= 10;
	elseif strcmp(t, 'mixed_cochlear_unk_ser_om') t= 11;
	elseif strcmp(t, 'mixed_poss_noise_om') t= 12;
	elseif strcmp(t, 'normal_ear') t= 13;
	elseif strcmp(t, 'otitis_media') t= 14;
	elseif strcmp(t, 'possible_brainstem_disorder') t= 15;
	elseif strcmp(t, 'possible_menieres') t= 16;
	elseif strcmp(t, 'retrocochlear_unknown') t= 17;
	else
	  error('clase %s descoñecida', t)
	end
	cl(i_fich,i)=t;
%  	printf('cl= %i\n', t);
%  	if i==2 exit end
  end
  fclose(f);
  
  if 2==i_fich
	continue
  end
  
  % busca repeticions
  for i=1:n_patrons(i_fich)
	for j=i+1:n_patrons(i_fich)
	  u=x(i_fich,i,:); v=x(i_fich,j,:);
	  if sqrt(sum((u - v).^2)) < 1e-3
		printf('patróns %i e %i iguais\n', i, j)
%  		for k=1:n_entradas
%  		  printf('%g ', u(k))
%  		end
%  		printf('\n')
%  		for k=1:n_entradas
%  		  printf('%g ', v(k))
%  		end
%  		printf('\n')
	  end
	end
  end
  
end
